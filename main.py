"""
Training script for Scene Graph Generation (or Scene Graph Prediction).

The script allows to reproduce the main experiments from our two papers:

[1] Boris Knyazev, Harm de Vries, Cătălina Cangea, Graham W. Taylor, Aaron Courville, Eugene Belilovsky.
Graph Density-Aware Losses for Novel Compositions in Scene Graph Generation. BMVC 2020. https://arxiv.org/abs/2005.08230

[2] Boris Knyazev, Harm de Vries, Cătălina Cangea, Graham W. Taylor, Aaron Courville, Eugene Belilovsky.
Generative Compositional Augmentations for Scene Graph Prediction. ICCV 2021. https://arxiv.org/abs/2007.05756

A large portion of this repo is based on https://github.com/rowanz/neural-motifs (MIT License).
For the paper [2], some GAN layers are based on https://github.com/google/sg2im (Apache-2.0 License).

Example to train IMP++ with GAN and GraphN scene graph perturbations:

    python main.py -ckpt ./data/VG/vg-faster-rcnn.tar -gan -largeD -loss dnorm -perturb graphn -vis_cond ./data/VG/features.hdf5

"""


from config import *
from dataloaders.visual_genome import VGDataLoader, VG
conf = ModelConfig()
VG.split = conf.split  # set VG, GQA or VTE split here to use as a global variable

from os.path import join
import pandas as pd
import time
import pickle

from sgg_models.rel_model_stanford import RelModelStanford
from lib.pytorch_misc import *
from lib.losses import node_losses, edge_losses
from lib.eval import val_epoch
from augment.gan import GAN
from augment.sg_perturb import SceneGraphPerturb

# Load VG data
train_loader, eval_loaders = VGDataLoader.splits(data_dir=conf.data,
                                                 batch_size=conf.batch_size,
                                                 num_workers=conf.num_workers,
                                                 num_gpus=conf.num_gpus,
                                                 is_cuda=conf.device=='cuda',
                                                 backbone=conf.backbone,
                                                 square_pad=True,
                                                 num_val_im=conf.val_size,
                                                 filter_non_overlap=conf.mode=='sgdet',
                                                 exclude_left_right=conf.exclude_left_right,
                                                 min_graph_size=conf.min_graph_size,
                                                 max_graph_size=conf.max_graph_size)

# Define SGG model
sgg_model = RelModelStanford(train_data=train_loader.dataset,
                             mode=conf.mode,
                             use_bias=conf.use_bias,
                             test_bias=conf.test_bias,
                             backbone=conf.backbone,
                             RELS_PER_IMG=conf.rels_per_img,
                             edge_model=conf.edge_model)
# Freeze the detector
for n, param in sgg_model.detector.named_parameters():
    param.requires_grad = False

gan = GAN(train_loader.dataset.ind_to_classes,
          train_loader.dataset.ind_to_predicates,
          n_ch=sgg_model.edge_dim,
          pool_sz=sgg_model.pool_sz,
          fmap_sz=sgg_model.fmap_sz,
          vis_cond=conf.vis_cond,
          losses=conf.ganlosses,
          init_embed=conf.init_embed,
          largeD=conf.largeD,
          device=conf.device,
          data_dir=train_loader.dataset.root) if conf.gan else None

checkpoint_path = None if conf.save_dir is None else join(conf.save_dir, 'vgrel.pth')
start_epoch, ckpt = load_checkpoint(conf, sgg_model, checkpoint_path, gan)
sgg_model.to(conf.device)
if conf.gan:
    gan.to(conf.device)
    if conf.perturb:
        set_seed(start_epoch + 1)  # to avoid repeating the same perturbations when reloaded from the checkpoint
        sgp = SceneGraphPerturb(method=conf.perturb,
                                embed_objs=gan.embed_objs,
                                subj_pred_obj_pairs=(train_loader.dataset.subj_pred_pairs,
                                                     train_loader.dataset.pred_obj_pairs),
                                obj_classes=train_loader.dataset.ind_to_classes,
                                triplet2str=train_loader.dataset.triplet2str,
                                L=conf.L, topk=conf.topk, alpha=conf.graphn_a,
                                uniform=conf.uniform, degree_smoothing=conf.degree_smoothing)

    if conf.wandb_log:
        wandb.watch(gan, log="all", log_freq=100 if conf.debug else 2000)

if conf.wandb_log:
    wandb.watch(sgg_model, log="all", log_freq=100 if conf.debug else 2000)


def train_batch(batch, verbose=False):
    set_mode(sgg_model, mode=conf.mode, is_train=True)

    res = sgg_model(batch.scatter())  # forward pass through an object detector and an SGG model

    # 1. Main SGG model object and relationship classification losses (L_cls)----------------------------------------------
    losses = node_losses(res.rm_obj_dists,   # predicted node labels (objects)
                         res.rm_obj_labels)  # predicted node labels (objects)

    loss, edges_fg, edges_bg = edge_losses(res.rel_dists,           # predicted edge labels (predicates)
                                           res.rel_labels[:, -1],   # ground truth edge labels (predicates)
                                           conf.loss,
                                           return_idx=True,
                                           loss_weights=(conf.alpha, conf.beta, conf.gamma))
    losses.update(loss)

    optimizer.zero_grad()
    loss = sum(losses.values())
    loss.backward()
    grad_clip(sgg_model, conf.clip, verbose)
    optimizer.step()
    # ------------------------------------------------------------------------------------------------------------------

    # 2. GAN-based updates----------------------------------------------------------------------------------------------
    if conf.gan:
        gan.train()
        # assume a single gpu!
        gt_boxes, gt_objects, gt_rels = batch[0][3].clone(), batch[0][4].clone(), batch[0][5].clone()

        if conf.perturb:
            # Scene Graph perturbations
            gt_objects_fake = sgp.perturb(gt_objects.clone(), gt_rels.clone()).clone()
        else:
            gt_objects_fake = gt_objects.clone()

        # Generate visual features conditioned on the SG
        fmaps = gan(gt_objects_fake,
                    sgg_model.get_scaled_boxes(gt_boxes, res.im_inds, res.im_sizes_org),
                    gt_rels)

        # Extract node,edge features from fmaps
        nodes_fake, edges_fake = sgg_model.node_edge_features(fmaps, res.rois, res.rel_inds[:, 1:], res.im_sizes)

        # Make SGG predictions for the node,edge features
        # In case of G update, detach generated features to avoid collaboration between the SGG model and G
        obj_dists_fake, rel_dists_fake = sgg_model.predict(nodes_fake if conf.attachG else nodes_fake.detach(),
                                                           edges_fake if conf.attachG else edges_fake.detach(),
                                                           res.rel_inds,
                                                           rois=res.rois,
                                                           im_sizes=res.im_sizes)

        # 2.1. Generator losses
        optimizer.zero_grad()
        G_optimizer.zero_grad()
        losses_G = {}
        losses_G.update(gan.loss(features_fake=nodes_fake, is_nodes=True, labels_fake=gt_objects_fake[:, -1]))
        losses_G.update(gan.loss(features_fake=edges_fake, labels_fake=res.rel_labels[:, -1]))
        losses_G.update(gan.loss(features_fake=fmaps, is_fmaps=True))
        for key in losses_G:
            losses_G[key] = conf.ganw * losses_G[key]

        if 'rec' in conf.ganlosses:
            sfx = '_rec'
            losses_G.update(node_losses(obj_dists_fake, gt_objects_fake[:, -1], sfx=sfx))
            losses_G.update(edge_losses(rel_dists_fake,
                                        res.rel_labels[:, -1],
                                        conf.loss,
                                        edges_fg, edges_bg,
                                        loss_weights=(conf.alpha, conf.beta, conf.gamma),
                                        sfx=sfx))

        if len(losses_G) > 0:
            loss = sum(losses_G.values())
            loss.backward()
            if 'rec' in conf.ganlosses:
                grad_clip(sgg_model, conf.clip, verbose)
                optimizer.step()
            G_optimizer.step()
            losses.update(losses_G)

        # 2.1. Discriminator losses
        D_optimizer.zero_grad()
        losses_D = {}
        losses_D.update(gan.loss(res.node_feat, nodes_fake, is_nodes=True, updateD=True, labels_fake=gt_objects_fake[:, -1],
                                 labels_real=gt_objects[:, -1]))
        losses_D.update(gan.loss(res.edge_feat, edges_fake, updateD=True,  labels_fake=res.rel_labels[:, -1]))
        losses_D.update(gan.loss(res.fmap, fmaps, updateD=True, is_fmaps=True))
        for key in losses_D:
            losses_D[key] = conf.ganw * losses_D[key]

        if len(losses_D) > 0:
            loss = sum(losses_D.values())
            loss.backward()
            D_optimizer.step()
            losses.update(losses_D)

    # ------------------------------------------------------------------------------------------------------------------

    # Compute for debugging purpose (not used for backprop)
    losses['total'] = sum(losses.values()).detach().data

    return pd.Series({x: tensor_item(y) for x, y in losses.items()})


def train_epoch(epoch_num):
    print('\nepoch %d, smallest lr %.3e\n' % (epoch_num, get_smallest_lr(optimizer)))
    sgg_model.train()
    tr = []
    start = time.time()
    for b, batch in enumerate(train_loader):

        tr.append(train_batch(batch, verbose=False))

        if conf.wandb_log:
            conf.wandb_log(tr[-1], step=sgg_model.global_batch_iter, prefix='loss/')

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1, sort=True).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval

            print(mn)

            time_eval_batch = time_per_batch

            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch ({:.1f}m/epoch including eval)\n".
                format(epoch_num, b, len(train_loader),
                       time_per_batch,
                       len(train_loader) * time_per_batch / 60,
                       len(train_loader) * time_eval_batch / 60))

            print('-----------', flush=True)

            start = time.time()

        sgg_model.global_batch_iter += 1

    return

optimizer, scheduler = get_optim(sgg_model, conf.lr * conf.num_gpus * conf.batch_size, conf, start_epoch, ckpt)
if conf.gan:
    G_optimizer, D_optimizer = get_optim_gan(gan, conf, start_epoch, ckpt)

print("\nTraining %s starts now!" % conf.mode.upper())

for epoch in range(start_epoch + 1, conf.num_epochs):

    scheduler.step(epoch)  # keep here for consistency with the paper
    train_epoch(epoch)

    other_states = {'epoch': epoch, 'global_batch_iter': sgg_model.global_batch_iter}
    if conf.gan:
        other_states.update({'gan': gan.state_dict(),
                             'G_optimizer': G_optimizer.state_dict(),
                             'D_optimizer': D_optimizer.state_dict() })
    save_checkpoint(sgg_model, optimizer, checkpoint_path, other_states)

    if epoch == start_epoch + 1 or (epoch % 5 == 0 and epoch < start_epoch + conf.num_epochs - 1):
        # evaluate only once in every 5 epochs since it's time consuming and evaluation is noisy
        for name, loader in eval_loaders.items():
            if name.startswith('val_'):
                val_epoch(conf.mode, sgg_model, loader, name,
                          train_loader.dataset.triplet_counts,
                          train_loader.dataset.triplet2str,
                          save_scores=conf.save_scores,
                          predicate_weight=conf.pred_weight,
                          train=train_loader.dataset,
                          wandb_log=conf.wandb_log)


# Evaluation on the test set here to make the pipeline complete
if conf.notest:
    print('evaluation on the test set is skipped due to the notest flag')
else:
    all_pred_entries = {}
    for name, loader in eval_loaders.items():
        if name.startswith('test_'):
            all_pred_entries[name] = val_epoch(conf.mode, sgg_model, loader, name,
                                               train_loader.dataset.triplet_counts,
                                               train_loader.dataset.triplet2str,
                                               is_test=True,
                                               save_scores=conf.save_scores,
                                               predicate_weight=conf.pred_weight,
                                               train=train_loader.dataset,
                                               wandb_log=conf.wandb_log)
    if conf.save_scores and conf.save_dir is not None:
        test_pred_f = join(conf.save_dir, 'test_predictions_%s.pkl' % conf.mode)
        print('saving test predictions to %s' % test_pred_f)
        with open(test_pred_f, 'wb') as f:
            pickle.dump(all_pred_entries, f)

print('done!')
