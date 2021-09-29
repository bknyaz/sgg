"""
Extracts node features from a pretrainend object detector.
Run this script before training a GAN.

Example (should take about 1 hour on a modern GPU):

    python extract_features.py -data ./data/ -ckpt ./data/VG/vg-faster-rcnn.tar -save_dir ./data/VG/

"""


from tqdm import tqdm
from config import *
from dataloaders.visual_genome import VGDataLoader, VG
conf = ModelConfig()
VG.split = conf.split  # set VG, GQA or VTE split here to use as a global variable

from sgg_models.rel_model_stanford import RelModelStanford
from lib.pytorch_misc import *

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

assert os.path.exists(conf.ckpt), ('need to load a pretrained detector', conf.ckpt)
start_epoch, _ = load_checkpoint(conf, sgg_model, None, None)
sgg_model.to(conf.device)
set_mode(sgg_model, mode=conf.mode, is_train=True)
sgg_model.detector.eval()

feat_file = os.path.join(conf.save_dir, 'features.hdf5')
with h5py.File(feat_file, 'a') as data_file:

    with torch.no_grad():
        for b, batch in enumerate(tqdm(train_loader)):
            res = sgg_model(batch.scatter())  # forward pass through an object detector and an SGG model
            gt_objects = batch[0][4].clone()


            for i, cls in enumerate(gt_objects[:, 1].data.cpu().numpy()):
                name = train_loader.dataset.ind_to_classes[cls]
                features = res.node_feat[i].data.cpu().numpy().astype(np.float32)

                if name not in data_file:
                    data_file.create_dataset(name, data=[features],
                                             maxshape=(None, *features.shape),
                                             chunks=(1, *features.shape),
                                             compression=4)
                else:
                    dset = data_file[name]
                    dset.resize(dset.shape[0] + 1, axis=0)
                    dset[-1, :] = features

    for i, name in enumerate(train_loader.dataset.ind_to_classes[1:]):
        print(i, name, data_file[name].shape)

print('done')
