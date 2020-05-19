"""
Training script for Scene Graph Generation

Based on https://github.com/rowanz/neural-motifs

"""


from config import *
from dataloaders.visual_genome import VGDataLoader, VG
conf = ModelConfig()
VG.split = conf.split  # set VG, GQA or VTE split here to use as a global variable

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from torch.nn.functional import cross_entropy as CE
from lib.pytorch_misc import *
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, eval_entry
import pickle
from lib.rel_model_stanford import RelModelStanford


EVAL_MODES = ['sgdet'] if conf.mode == 'sgdet' else ['predcls', 'sgcls']
assert conf.mode in EVAL_MODES, (conf.mode, 'other modes not supported')

train, val_splits = VG.splits(data_dir=conf.data,
                              num_val_im=conf.val_size,
                              min_graph_size=conf.min_graph_size,
                              max_graph_size=conf.max_graph_size,
                              mrcnn=conf.detector == 'mrcnn',
                              filter_non_overlap=conf.mode == 'sgdet',
                              exclude_left_right=conf.exclude_left_right)

train_loader, val_loaders = VGDataLoader.splits(train, val_splits,
                                               mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)
val_loader, val_loader_zs, test_loader, test_loader_zs = val_loaders

detector = RelModelStanford(train_data=train,
                            num_gpus=conf.num_gpus,
                            mode=conf.mode,
                            use_bias=conf.use_bias,
                            test_bias=conf.test_bias,
                            detector_model=conf.detector,
                            RELS_PER_IMG=conf.rels_per_img)

# Freeze the detector
for n, param in detector.detector.named_parameters():
    param.requires_grad = False

print(print_para(detector), flush=True)
# print(detector)

checkpoint_name = 'vgrel.pth'
checkpoint_path = os.path.join(conf.save_dir, checkpoint_name)

start_epoch = -1
detector.global_batch_iter = 0  # for wandb
ckpt = None

checkpoint_path_load = checkpoint_path if os.path.exists(checkpoint_path) \
    else (conf.ckpt if len(conf.ckpt) > 0 else None)

if checkpoint_path_load is not None:
    print("Loading EVERYTHING from %s" % checkpoint_path_load)
    ckpt = torch.load(checkpoint_path_load)
    success = optimistic_restore(detector, ckpt['state_dict'])

    if success and os.path.exists(checkpoint_path):  # vgrel.pth
        # If there's already a checkpoint in the save_dir path, assume we should load it and continue
        # Useful to restart the job with exactly the same parameters
        start_epoch = ckpt['epoch']
        detector.global_batch_iter = ckpt['global_batch_iter']


detector.to(conf.device)

if conf.wandb_log:
    wandb.watch(detector, log="all", log_freq=100 if conf.debug else 2000)


def train_batch(b, verbose=False):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
    :return:
    """
    set_mode(detector, mode=conf.mode, is_train=True, conf=conf)

    optimizer.zero_grad()
    res = detector.forward_parallel(b)
    losses = {'obj_loss': CE(res.rm_obj_dists, res.rm_obj_labels)}

    idx_fg = torch.nonzero(res.rel_labels[:, -1] > 0).data.view(-1)
    idx_bg = torch.nonzero(res.rel_labels[:, -1] == 0).data.view(-1)
    M_FG = len(idx_fg)
    M_BG = len(idx_bg)
    M = len(res.rel_dists)

    loss = CE(res.rel_dists, res.rel_labels[:, -1], reduce=False)

    if conf.loss == 'baseline':

        assert conf.alpha == conf.beta == 1, ('wrong loss is used', conf.alpha, conf.beta)
        loss = conf.lam * (loss / M)  # weight all edges by the same value (divide by M to compute average below)
        losses['rel_loss'] = loss.sum()  # loss is averaged over all FG and BG edges

    elif conf.loss in ['dnorm', 'dnorm-fgbg']:

        edge_weights = torch.ones(M, device=conf.device)

        if M_FG > 0:
            edge_weights[idx_fg] = float(conf.alpha) / M_FG   # weight for FG edges (alpha/M_FG instead of 1/M as in the baseline)

        if conf.loss == 'dnorm':
            # conf.alpha = conf.beta = 1 in our hyperparameter-free loss
            if M_BG > 0 and M_FG > 0:
                edge_weights[idx_bg] = float(conf.beta) / M_FG   # weight for BG edges (beta/M_FG instead of 1/M as in the baseline)
        else:
            if M_BG > 0:
                edge_weights[idx_bg] = float(conf.beta) / M_BG   # weight for BG edges (beta/M_BG instead of 1/M as in the baseline)

        loss = conf.gamma * loss * torch.autograd.Variable(edge_weights)
        losses['rel_loss'] = loss.sum()
    else:
        raise NotImplementedError(conf.loss)

    loss_all = sum(losses.values())
    loss_all.backward()
    grad_clip(detector, conf.clip, verbose)
    optimizer.step()  # update Rel and Obj models

    # Compute and log for debugging purposes, but do not use for backprop
    losses['total'] = loss_all.detach().data
    if len(idx_fg) > 0:
        losses['rel_loss_fg'] = loss[idx_fg].sum().detach().data  # average loss for each FG edge
    if len(idx_bg) > 0:
        losses['rel_loss_bg'] = loss[idx_bg].sum().detach().data  # average loss for each BG edge

    res = pd.Series({x: tensor_item(y) for x, y in losses.items()})  # data[0]
    return res


def train_epoch(epoch_num):
    print('\nepoch %d, smallest lr %f\n' % (epoch, get_smallest_lr(optimizer)))
    detector.train()
    tr = []
    start = time.time()
    for b, batch in enumerate(train_loader):

        tr.append(train_batch(batch, verbose=b % (conf.print_interval * 20) == 0))

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1, sort=True).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval

            print(mn)

            if conf.wandb_log:
                conf.wandb_log(mn.to_dict(), step=detector.global_batch_iter, prefix='loss/')

            time_eval_batch = time_per_batch

            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch ({:.1f}m/epoch including eval)\n".
                format(epoch_num, b, len(train_loader),
                       time_per_batch,
                       len(train_loader) * time_per_batch / 60,
                       len(train_loader) * time_eval_batch / 60))

            print('-----------', flush=True)

            start = time.time()

        detector.global_batch_iter += 1

    rez = pd.concat(tr, axis=1, sort=True)
    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)

    return rez


def val_batch(batch_num, b, evaluator, eval_m, val_dataset, evaluator_list, evaluator_multiple_preds_list):

    if conf.detector == 'mrcnn':
        scale = 1.
        box_threshs = [0.2, 0.05, 0.01]
    else:
        scale = BOX_SCALE / IM_SCALE
        box_threshs = [None]

    pred_entries = []
    for box_score_thresh in box_threshs:
        detector.set_box_score_thresh(box_score_thresh)
        try:
            det_res = detector.forward_parallel(b)  # keep as it was in the original code

            if conf.num_gpus == 1:
                det_res = [det_res]

            for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
                gt_entry = {
                    'gt_classes': val_dataset.gt_classes[batch_num + i].copy(),
                    'gt_relations': val_dataset.relationships[batch_num + i].copy(),
                    'gt_boxes': val_dataset.gt_boxes[batch_num + i].copy(),
                }

                pred_entry = {
                    'pred_boxes': boxes_i * scale,
                    'pred_classes': objs_i,
                    'pred_rel_inds': rels_i,
                    'obj_scores': obj_scores_i,
                    'rel_scores': pred_scores_i,  # hack for now.
                }
                pred_entries.append(pred_entry)

                for sfx in ['', '_nogc']:
                    evaluator[eval_m + sfx].evaluate_scene_graph_entry(
                        gt_entry,
                        pred_entry
                    )

                if evaluator_list is not None and len(evaluator_list) > 0:
                    eval_entry(eval_m, gt_entry, pred_entry,
                               evaluator_list, evaluator_multiple_preds_list)

            return pred_entries

        except ValueError as e:
            print('no objects or relations found'.upper(), e, b[0][-1], 'trying a smaller threshold')


def val_epoch(loader, name, n_batches=-1, is_test=False):
    print('\nEvaluate %s %s triplets' % (name.upper(), 'test' if is_test else 'val'))
    detector.eval()
    evaluator, all_pred_entries, all_metrics = {}, {}, []
    with NO_GRAD():
        for eval_m in EVAL_MODES:
            if eval_m == 'sgdet' and name.find('val_') >= 0:
                continue  # skip for validation, because it takes a lot of time

            print('\nEvaluating %s...' % eval_m.upper())

            evaluator[eval_m] = BasicSceneGraphEvaluator(eval_m)  # graph constrained evaluator
            evaluator[eval_m + '_nogc'] = BasicSceneGraphEvaluator(eval_m, multiple_preds=True,    # graph unconstrained evaluator
                                                                   per_triplet=name not in ['val_zs', 'test_zs'],
                                                                   triplet_counts=train.triplet_counts,
                                                                   triplet2str=train_loader.dataset.triplet2str)

            # for calculating recall of each relationship except no relationship
            evaluator_list, evaluator_multiple_preds_list = [], []
            if name not in ['val_zs', 'test_zs'] and name.find('val_') < 0:
                for index, name_s in enumerate(train.ind_to_predicates):
                    if index == 0:
                        continue
                    evaluator_list.append((index, name_s, BasicSceneGraphEvaluator.all_modes()))
                    evaluator_multiple_preds_list.append(
                        (index, name_s, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))


            set_mode(detector, mode=eval_m, is_train=False, conf=conf, verbose=True)


            # For all val/test batches
            all_pred_entries[eval_m] = []
            for val_b, batch in enumerate(tqdm(loader)):
                pred_entry = val_batch(conf.num_gpus * val_b, batch, evaluator, eval_m, loader.dataset, evaluator_list, evaluator_multiple_preds_list)
                if not conf.nosave:
                    all_pred_entries[eval_m].extend(pred_entry)

                if n_batches > -1 and val_b + 1 >= n_batches:
                    break


            evaluator[eval_m].print_stats()
            evaluator[eval_m + '_nogc'].print_stats()

            mean_recall = mean_recall_mp = None
            if len(evaluator_list) > 0:
                # Compute Mean Recall Results
                mean_recall = calculate_mR_from_evaluator_list(evaluator_list, eval_m, save_file=None)
                mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, eval_m,
                                                                  multiple_preds=True, save_file=None)

            if not conf.wandb_log:
                continue

            # Log using WANDB
            eval_gc = evaluator[eval_m].result_dict
            eval_no_gc = evaluator[eval_m + '_nogc'].result_dict
            results_dict = {}
            for eval_, mean_eval, sfx in zip([eval_gc, eval_no_gc], [mean_recall, mean_recall_mp], ['GC', 'NOGC']):
                for k, v in eval_[eval_m + '_recall'].items():
                    all_metrics.append(np.mean(v))
                    results_dict['%s/%s_R@%i_%s' % (eval_m, name, k, sfx)] = np.mean(v)
                if mean_eval:
                    for k, v in mean_eval.items():
                        results_dict['%s/%s_m%s_%s' % (eval_m, name, k, sfx)] = np.mean(v)

            # Per triplet metrics
            if name not in ['val_zs', 'test_zs']:
                for case in ['', '_norm']:
                    for k, v in eval_no_gc[eval_m + '_recall_triplet' + case].items():
                        results_dict['%s/%s_R@%i_triplet%s' % (eval_m, name, k, case)] = v
                    for metric in ['meanrank', 'medianrank'] + (['medianrankclass'] if case == '' else []):
                        results_dict['%s/%s_%s_triplet%s' % (eval_m, name, metric, case)] = \
                            eval_no_gc[eval_m + ('_%s_triplet' % metric) + case]

            conf.wandb_log(results_dict, step=detector.global_batch_iter,
                           is_summary=True, log_repeats=5 if is_test else 1)


    if conf.wandb_log:
        conf.wandb_log({'avg/%s_R' % (name): np.mean(all_metrics)}, step=detector.global_batch_iter,
                       is_summary=True, log_repeats=5 if is_test else 1)

    return all_pred_entries


print("\nTraining %s starts now!" % conf.mode.upper())
optimizer, scheduler = get_optim(detector, conf.lr * conf.num_gpus * conf.batch_size, conf, start_epoch, ckpt)

for epoch in range(start_epoch + 1, conf.num_epochs):
    rez = train_epoch(epoch)
    save_checkpoint(detector, optimizer, conf.save_dir, checkpoint_name,
                    {'epoch': epoch, 'global_batch_iter': detector.global_batch_iter})

    if epoch == start_epoch + 1 or (epoch % 5 == 0 and epoch < start_epoch + conf.num_epochs - 1):
        # evaluate only once in every 5 epochs since it's time consuming and evaluation is noisy
        for loader, name in list(zip([val_loader_zs, val_loader], ['val_zs', 'val_all_large'])):
            val_epoch(loader, name)

    detector.global_batch_iter += 1  # to increase the counter for wandb

    print('\nscheduler before step, epoch %d, smallest lr %f' % (epoch, get_smallest_lr(optimizer)))
    scheduler.step(epoch)
    print('scheduler after step, epoch %d, smallest lr %f\n' % (epoch, get_smallest_lr(optimizer)))


# Evaluation on the test set here to make the pipeline complete
if not conf.notest:
    all_pred_entries = {}
    for loader, name in list(zip([test_loader_zs, test_loader], ['test_zs', 'test_all_large'])):
        all_pred_entries[name] = val_epoch(loader, name, is_test=True)

    if conf.nosave or len(conf.save_dir) == 0:
        print('saving test predictions is ommitted due to the nosave argument or save_dir not specified', conf.save_dir)
    else:
        test_pred_f = os.path.join(conf.save_dir, 'test_predictions_%s.pkl' % conf.mode)
        print('saving test predictions to %s' % test_pred_f)
        with open(test_pred_f, 'wb') as f:
            pickle.dump(all_pred_entries, f)
else:
    print('evaluation on the test set is skipped due to the notest flag')

print('done!')
