"""
File that involves dataloaders for the Visual Genome dataset.
"""

import json
import os

import h5py
import numpy as np
import torch
import copy
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from dataloaders.blob import Blob
from lib.pytorch_misc import bbox_overlaps
from dataloaders.image_transforms import SquarePad
from dataloaders.vtranse import VTESplit
import dataloaders.gqa as gqa
from collections import defaultdict
from config import BOX_SCALE, IM_SCALE


class VG(Dataset):

    split='stanford'  # 'stanford', 'vte', 'gqa'

    # to avoid reading files several times
    filenames = None
    train_sgs = None
    val_sgs = None

    def __init__(self, mode, data_dir, filter_empty_rels=True, num_im=-1, num_val_im=5000,
                 filter_duplicate_rels=True, filter_non_overlap=True,
                 min_graph_size=-1, max_graph_size=-1,
                 torch_detector=False, n_shots=-1,
                 square_pad=True,
                 training_triplets=None, exclude_left_right=False):
        """0
        Torch dataset for VisualGenome
        :param mode: Must be train, test, or val
        :param roidb_file:  HDF5 containing the GT boxes, classes, and relationships
        :param dict_file: JSON Contains mapping of classes/relationships to words
        :param image_file: HDF5 containing image filenames
        :param filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
        :param filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
        :param num_im: Number of images in the entire dataset. -1 for all images.
        :param num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        :param proposal_file: If None, we don't provide proposals. Otherwise file for where we get RPN
            proposals
        """

        # print(mode, VG.split, data_dir, num_im, num_val_im, torch_detector, min_graph_size, max_graph_size, n_shots)

        assert mode in ('test', 'train', 'val'), '%s mode not recognized' % mode
        self.mode = mode
        self.max_graph_size = max_graph_size if mode == 'train' else -1
        self.min_graph_size = min_graph_size if mode == 'train' else -1
        self.filter_non_overlap = filter_non_overlap
        self.filter_duplicate_rels = filter_duplicate_rels and self.mode == 'train'
        self.n_shots = n_shots
        assert VG.split in ['stanford', 'vte', 'gqa'], ('invalid split', VG.split)

        if training_triplets:
            assert mode in ['val', 'test'], mode

        if VG.split == 'stanford':
            data_name = 'VG'
            self.roidb_file = os.path.join(data_dir, data_name, 'stanford_filtered', 'VG-SGG.h5')
            self.dict_file = os.path.join(data_dir, data_name, 'stanford_filtered', 'VG-SGG-dicts.json')
            self.image_file = os.path.join(data_dir, data_name, 'stanford_filtered', 'image_data.json')
            self.images_dir = os.path.join(data_dir, data_name, 'VG_100K')
            self.split_mask, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
                self.roidb_file, self.mode, num_im, num_val_im=num_val_im,
                filter_empty_rels=filter_empty_rels,
                min_graph_size=self.min_graph_size,
                max_graph_size=self.max_graph_size,
                filter_non_overlap=self.filter_non_overlap and self.is_train,
                training_triplets=training_triplets,
                random_subset=False,
                filter_zeroshots=True,
                n_shots=n_shots
            )
        elif VG.split == 'vte':
            data_name = 'VG'
            self.images_dir = os.path.join(data_dir, data_name, 'VG_100K')
            vte = VTESplit(os.path.join(data_dir, data_name, 'vtranse', 'vg1_2_meta.h5'), mode=self.mode)
            self.split_mask, self.gt_boxes, self.gt_classes, self.relationships = vte.load_graphs(
                num_im,
                num_val_im=num_val_im,
                filter_empty_rels=filter_empty_rels,
                min_graph_size=self.min_graph_size,
                max_graph_size=self.max_graph_size,
                training_triplets=training_triplets,
                random_subset=False,
                filter_zeroshots=True
            )

        elif VG.split == 'gqa':
            data_name = 'GQA'
            self.images_dir = os.path.join(data_dir, 'VG/VG_100K')
            # Load the JSON containing the SGs
            f_mode = mode
            if mode == 'val':
                f_mode = 'train'  # we are using the last 5k training SGs for validation
            elif mode == 'test':
                f_mode = 'val'  # GQA has no public test SGs, so use the val set instead

            img_list_file = os.path.join(data_dir, data_name, '%s_images.json' % f_mode)

            if os.path.isfile(img_list_file):
                print('Loading GQA-%s image ids...' % mode)
                with open(img_list_file, 'r') as f:
                    self.image_ids = json.load(f)
            else:
                # Use only images having question-answer pairs in the balanced split
                print('Loading GQA-%s questions...' % mode)
                with open(os.path.join(data_dir, data_name, '%s_balanced_questions.json' % f_mode), 'rb') as f:
                    Q_dict = json.load(f)
                self.image_ids = set()
                for v in Q_dict.values():
                    self.image_ids.add(v['imageId'])
                with open(img_list_file, 'w') as f:
                    json.dump(list(self.image_ids), f)

                del Q_dict

            self.image_ids = sorted(list(self.image_ids))  # sort to make it consistent for different runs

            self.filenames = gqa.load_image_filenames(self.image_ids, mode, self.images_dir)

            if VG.train_sgs is None:
                print('Loading GQA-%s scene graphs...' % mode)
                with open(os.path.join(data_dir, data_name, 'sceneGraphs/train_sceneGraphs.json'), 'rb') as f:
                    VG.train_sgs = json.load(f)
                with open(os.path.join(data_dir, data_name, 'sceneGraphs/val_sceneGraphs.json'), 'rb') as f:
                    VG.val_sgs = json.load(f)
            train_sgs, val_sgs = VG.train_sgs, VG.val_sgs

            (self.ind_to_classes, self.ind_to_predicates,
             self.classes_to_ind, self.predicates_to_ind) = gqa.load_info(train_sgs, val_sgs)

            (self.split_mask, self.gt_boxes,
             self.gt_classes, self.relationships) = gqa.load_graphs(
                train_sgs if f_mode == 'train' else val_sgs,
                self.image_ids,
                self.classes_to_ind, self.predicates_to_ind,
                num_val_im=num_val_im,
                mode=mode,
                training_triplets=training_triplets,
                min_graph_size=self.min_graph_size,
                max_graph_size=self.max_graph_size,
                random_subset=False,
                filter_empty_rels=filter_empty_rels,
                filter_zeroshots=True,
                exclude_left_right=exclude_left_right
            )

            del train_sgs, val_sgs, self.image_ids  # force to clean RAM

        else:
            raise NotImplementedError(VG.split)

        self.root = os.path.join(data_dir, data_name)

        if VG.split == 'stanford':
            self.filenames = load_image_filenames(self.image_file, self.images_dir) if VG.filenames is None else VG.filenames
            self.ind_to_classes, self.ind_to_predicates = load_info(self.dict_file)
        elif VG.split == 'vte':
            self.filenames = vte.load_image_filenames(self.images_dir)
            self.ind_to_classes, self.ind_to_predicates = vte.load_info()
            vte.close()

        if VG.filenames is None:
            VG.filenames = self.filenames

        # if self.mode == 'train':
        #     print('\nind_to_classes', len(self.ind_to_classes), self.ind_to_classes)
        #     print('\nind_to_predicates', len(self.ind_to_predicates), self.ind_to_predicates, '\n')

        self.triplet_counts = {}
        # c = 0
        N_total, M_FG_total, M_BG_total, n_obj_lst, fg_lst, sp_lst  = 0, 0, 0, [], [], []
        for im in range(len(self.gt_classes)):
            n_obj = len(self.gt_classes[im])
            n_obj_lst.append(n_obj)
            fg_lst.append(len(filter_dups(self.relationships[im])))
            sp_lst.append(100 * float(fg_lst[-1]) / (n_obj * (n_obj - 1)))
            N_total += n_obj
            M_FG_total += fg_lst[-1]
            M_BG_total += n_obj * (n_obj - 1)
            for rel_ind, tri in enumerate(self.relationships[im]):
                o1, o2, R = tri
                tri_str = '{}_{}_{}'.format(self.gt_classes[im][o1], R, self.gt_classes[im][o2])

                if training_triplets:
                    if isinstance(training_triplets, dict):
                        assert tri_str in training_triplets and training_triplets[tri_str] <= self.n_shots, (mode, len(training_triplets), tri_str, training_triplets[tri_str], self.n_shots)

                if tri_str not in self.triplet_counts:
                    self.triplet_counts[tri_str] = 0

                self.triplet_counts[tri_str] += 1

        n_samples = len(self.gt_classes)
        # self.triplets = list(self.triplet_counts.keys())
        counts = list(self.triplet_counts.values())

        if mode == 'train':
            self.subj_pred_pairs, self.pred_obj_pairs = {}, {}
            for im in range(len(self.gt_classes)):
                for rel_ind, tri in enumerate(self.relationships[im]):
                    o1, o2, R = tri
                    tri_str = '{}_{}_{}'.format(self.gt_classes[im][o1], R, self.gt_classes[im][o2])
                    pair = '{}_{}'.format(self.gt_classes[im][o1], R)  # man wearing
                    if pair not in self.subj_pred_pairs:
                        self.subj_pred_pairs[pair] = {}
                    self.subj_pred_pairs[pair][self.gt_classes[im][o2]] = self.triplet_counts[tri_str]

                    pair = '{}_{}'.format(R, self.gt_classes[im][o2])  # on surfboard
                    if pair not in self.pred_obj_pairs:
                        self.pred_obj_pairs[pair] = {}
                    self.pred_obj_pairs[pair][self.gt_classes[im][o1]] = self.triplet_counts[tri_str]

            print('subj_pred_pairs, pred_obj_pairs', len(self.subj_pred_pairs), len(self.pred_obj_pairs))

        print('{} images, {} triplets ({} unique triplets)'.format(
            # VG.split,
            # mode,
            len(self.gt_classes),
            # np.sum(self.split_mask),
            # len(self.split_mask),
            np.sum(counts),  # total count
            len(self.triplet_counts)))  # unique
            # np.min(counts), np.max(counts)))

        def stats(x):
            return 'min={:.1f}, max={:.1f}, mean={:.1f}, std={:.1f}'.format(np.min(x), np.max(x), np.mean(x), np.std(x))

        print('Stats: {} objects ({}), {} FG edges ({}), {} BG edges ({:.2f} avg), graph density {}'.format(
            N_total,
            str(stats(n_obj_lst)),
            M_FG_total,
            str(stats(fg_lst)),
            M_BG_total,
            M_BG_total / n_samples,
            str(stats(sp_lst))))

        assert len(self.split_mask) == len(self.filenames), (len(self.split_mask), len(self.filenames))

        self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]

        # if self.mode == 'train':
        #     print('example of triplets')
        #     for tri in list(self.triplet_counts.keys())[:5]:
        #         print(tri, self.triplet2str(tri), self.triplet_counts[tri])

        self.rpn_rois = None

        self.torch_detector = torch_detector

        tform = [SquarePad()] if square_pad else []
        if torch_detector:
            tform += [ToTensor()]
        else:
            tform += [
                Resize(IM_SCALE),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]

        self.transform_pipeline = Compose(tform)


    def triplet2str(self, triplet):
        o1, R, o2 = triplet.split('_')
        try:
            return '_'.join((self.ind_to_classes[int(o1)], self.ind_to_predicates[int(R)], self.ind_to_classes[int(o2)]))
        except:
            print(triplet, len(self.ind_to_classes), len(self.ind_to_predicates))
            raise

    @property
    def coco(self):
        """
        :return: a Coco-like object that we can use to evaluate detection!
        """
        anns = []
        for i, (cls_array, box_array) in enumerate(zip(self.gt_classes, self.gt_boxes)):
            for cls, box in zip(cls_array.tolist(), box_array.tolist()):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': i,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'ayy lmao'},
            'images': [{'id': i} for i in range(self.__len__())],
            'categories': [{'supercategory': 'person',
                               'id': i, 'name': name} for i, name in enumerate(self.ind_to_classes) if name != '__background__'],
            'annotations': anns,
        }
        fauxcoco.createIndex()
        return fauxcoco

    @property
    def is_train(self):
        return self.mode.startswith('train')

    @classmethod
    def splits(cls, *args, **kwargs):
        """ Helper method to generate splits of the dataset"""

        print('\nLoading the %s split of Visual Genome' % VG.split.upper())
        print('\nTRAIN DATASET')
        train = cls('train', *args, **kwargs)

        if ('min_graph_size' in kwargs and kwargs['min_graph_size'] > -1) or \
                ('max_graph_size' in kwargs and kwargs['max_graph_size'] > -1) \
                or train.filter_non_overlap:
            print('loading the original training split first')
            kwargs['min_graph_size'] = -1
            kwargs['max_graph_size'] = -1
            kwargs['filter_non_overlap'] = False
            train_orig = cls('train', *args, **kwargs)
            train.triplet_counts = train_orig.triplet_counts
        else:
            train_orig = train

        eval_splits = {}
        print('\nVAL DATASET (ZERO-SHOTS)')
        eval_splits['val_zs'] = cls('val', *args, **kwargs, training_triplets=set(list(train_orig.triplet_counts.keys())))

        print('\nVAL DATASET (ALL-SHOTS)')
        eval_splits['val_alls'] = cls('val', *args, **kwargs)

        print('\nTEST DATASET (ZERO SHOTS)')
        eval_splits['test_zs'] = cls('test', *args, **kwargs,
                      training_triplets=set(list(train_orig.triplet_counts.keys()) + list(eval_splits['val_alls'].triplet_counts.keys())))
        n_img = {'stanford': 4519, 'vte': 653, 'gqa': 6418}
        assert len(eval_splits['test_zs']) == n_img[VG.split], (len(eval_splits['test_zs']), VG.split)

        if VG.split == 'stanford':
            # TODO: extend for other splits like GQA

            print('\nTEST DATASET (10-SHOTS)')
            triplet_counts = copy.deepcopy(train_orig.triplet_counts)
            for k, v in eval_splits['val_alls'].triplet_counts.items():
                if k not in triplet_counts:
                    triplet_counts[k] = 0
                triplet_counts[k] += v
            eval_splits['test_10s'] = cls('test', *args, **kwargs, training_triplets=triplet_counts, n_shots=10)
            n_img = {'stanford': 9602, 'vte': -1, 'gqa': -1}
            if VG.split == 'stanford':
                assert len(eval_splits['test_10s']) == n_img[VG.split], (len(eval_splits['test_10s']), VG.split)

            print('\nTEST DATASET (100-SHOTS)')
            eval_splits['test_100s'] = cls('test', *args, **kwargs, training_triplets=triplet_counts, n_shots=100)
            n_img = {'stanford': 16528, 'vte': -1, 'gqa': -1}
            if VG.split == 'stanford':
                assert len(eval_splits['test_100s']) == n_img[VG.split], (len(eval_splits['test_100s']), VG.split)

        print('\nTEST DATASET (ALL-SHOTS)')
        eval_splits['test_alls'] = cls('test', *args, **kwargs)
        n_img = {'stanford': 26446, 'vte': 25851, 'gqa': 10055}
        assert len(eval_splits['test_alls']) == n_img[VG.split], (len(eval_splits['test_alls']), VG.split)

        return train, eval_splits


    def __getitem__(self, index):
        image_unpadded = Image.open(os.path.join(self.images_dir, self.filenames[index])).convert('RGB')

        # Optionally flip the image if we're doing training
        flipped = self.is_train and np.random.random() > 0.5
        gt_boxes = self.gt_boxes[index].copy()
        w, h = image_unpadded.size

        if self.torch_detector:
            im_scale = max(w, h)
            box_scale = im_scale
        else:
            im_scale = IM_SCALE
            box_scale = BOX_SCALE

        box_scale_factor = box_scale / max(w, h)

        if VG.split in ['vte', 'gqa']:
            gt_boxes = gt_boxes * box_scale_factor  # multiply bbox by this number to make them BOX_SCALE
        elif self.torch_detector:
            # make the boxes at image scale
            s = BOX_SCALE / max(w, h)
            gt_boxes = gt_boxes / s

        if self.torch_detector:
            assert box_scale_factor == 1, box_scale_factor

        # crop boxes that are too large. This seems to be only a problem for image heights, but whatevs
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clip(
            None, box_scale / max(image_unpadded.size) * image_unpadded.size[1])
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clip(
            None, box_scale / max(image_unpadded.size) * image_unpadded.size[0])


        if VG.split in ['vte', 'gqa']:
            # width, height can become zero after clipping (need to double-check why)
            ind_zero = (gt_boxes[:, 2] - gt_boxes[:, 0]) == 0 & (gt_boxes[:, 0] > 0)  # x1 == x2 and x1 > 0
            gt_boxes[ind_zero, 0] -= 1
            ind_zero = (gt_boxes[:, 3] - gt_boxes[:, 1]) == 0 & (gt_boxes[:, 1] > 0)  # y1 == y2 and y1 > 0
            gt_boxes[ind_zero, 1] -= 1


        if flipped:
            scaled_w = int(box_scale_factor * float(w))
            # print("Scaled w is {}".format(scaled_w))
            image_unpadded = image_unpadded.transpose(Image.FLIP_LEFT_RIGHT)
            gt_boxes[:, [0, 2]] = scaled_w - gt_boxes[:, [2, 0]]

        img_scale_factor = im_scale / max(w, h)
        if h > w:
            im_size = (im_scale, int(w * img_scale_factor), img_scale_factor)
        elif h < w:
            im_size = (int(h * img_scale_factor), im_scale, img_scale_factor)
        else:
            im_size = (im_scale, im_scale, img_scale_factor)

        gt_rels = self.relationships[index].copy()
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.mode == 'train'
            gt_rels = filter_dups(gt_rels)

        entry = {
            'img': self.transform_pipeline(image_unpadded),
            'img_size': im_size,
            'gt_boxes': gt_boxes,
            'gt_classes': self.gt_classes[index].copy(),
            'gt_relations': gt_rels,
            'scale': im_scale / box_scale,  # Multiply the boxes by this.
            'index': index,
            'flipped': flipped,
            'fn': self.filenames[index],
        }

        if self.rpn_rois is not None:
            entry['proposals'] = self.rpn_rois[index]

        assertion_checks(entry)
        return entry

    def __len__(self):
        return len(self.filenames)

    @property
    def num_predicates(self):
        return len(self.ind_to_predicates)

    @property
    def num_classes(self):
        return len(self.ind_to_classes)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MISC. HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def assertion_checks(entry):
    im_size = tuple(entry['img'].size())
    if len(im_size) != 3:
        raise ValueError("Img must be dim-3")

    c, h, w = entry['img'].size()
    if c != 3:
        raise ValueError("Must have 3 color channels")

    num_gt = entry['gt_boxes'].shape[0]
    if entry['gt_classes'].shape[0] != num_gt:
        raise ValueError("GT classes and GT boxes must have same number of examples")

    assert (entry['gt_boxes'][:, 2] >= entry['gt_boxes'][:, 0]).all()
    assert (entry['gt_boxes'] >= -1).all()


def load_image_filenames(image_file, image_dir):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    :param image_file: JSON file. Elements contain the param "image_id".
    :param image_dir: directory where the VisualGenome images are located
    :return: List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(image_dir, basename)
        if os.path.exists(filename):  # can comment for faster loading
            fns.append(basename)  # add basename only to save RAM
    assert len(fns) == 108073, len(fns)
    return fns


def load_graphs(graphs_file, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True, min_graph_size=-1, max_graph_size=-1,
                filter_non_overlap=False, training_triplets=None, random_subset=False, filter_zeroshots=True, n_shots=-1):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    :param graphs_file: HDF5
    :param mode: (train, val, or test)
    :param num_im: Number of images we want
    :param num_val_im: Number of validation images
    :param filter_empty_rels: (will be filtered otherwise.)
    :param filter_non_overlap: If training, filter images that dont overlap.
    :return: image_index: numpy array corresponding to the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground 
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of 
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    with h5py.File(graphs_file, 'r') as roi_h5:
        data_split = roi_h5['split'][:]
        split = 2 if mode == 'test' else 0
        split_mask = data_split == split

        # Filter out images without bounding boxes
        split_mask &= roi_h5['img_to_first_box'][:] >= 0
        if filter_empty_rels:
            split_mask &= roi_h5['img_to_first_rel'][:] >= 0

        image_index = np.where(split_mask)[0]
        if num_im > -1:
            image_index = image_index[:num_im]
        if num_val_im > 0:
            if mode in ['val']:  # , 'test' for faster preliminary evaluation on the test set
                image_index = image_index[:num_val_im]
            elif mode == 'train':
                image_index = image_index[num_val_im:]


        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[image_index] = True

        # Get box information
        all_labels = roi_h5['labels'][:, 0]
        all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # will index later
        assert np.all(all_boxes[:, :2] >= 0)  # sanity check
        assert np.all(all_boxes[:, 2:] > 0)  # no empty box

        # convert from xc, yc, w, h to x1, y1, x2, y2
        all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
        all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

        im_to_first_box = roi_h5['img_to_first_box'][split_mask]
        im_to_last_box = roi_h5['img_to_last_box'][split_mask]
        im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
        im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

        # load relation labels
        _relations = roi_h5['relationships'][:]
        _relation_predicates = roi_h5['predicates'][:, 0]
        assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
        assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    relationships = []
    for i in range(len(image_index)):
        boxes_i = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
        gt_classes_i = all_labels[im_to_first_box[i]:im_to_last_box[i] + 1]

        if min_graph_size > -1 and len(gt_classes_i) <= min_graph_size:  # 0-min_graph_size will be excluded
            split_mask[image_index[i]] = 0
            continue

        if max_graph_size > -1 and len(gt_classes_i) > max_graph_size:  # max_graph_size+1-Inf will be excluded
            split_mask[image_index[i]] = 0
            continue

        ind_zs = []
        if im_to_first_rel[i] >= 0:
            predicates = _relation_predicates[im_to_first_rel[i]:im_to_last_rel[i] + 1]
            obj_idx = _relations[im_to_first_rel[i]:im_to_last_rel[i] + 1] - im_to_first_box[i]
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))

            if training_triplets:
                n = len(rels)
                if n > 0:
                    if random_subset:
                        ind_zs = np.random.permutation(n)[:int(np.round( n / 15. ))]
                    else:
                        for rel_ind, tri in enumerate(rels):
                            o1, o2, R = tri
                            tri_str = '{}_{}_{}'.format(gt_classes_i[o1], R, gt_classes_i[o2])

                            if isinstance(training_triplets, dict):
                                assert n_shots > 0, n_shots
                                if tri_str in training_triplets:
                                    if (n_shots == 10 and training_triplets[tri_str] >= 1 and training_triplets[tri_str] <= n_shots) \
                                            or (n_shots == 100 and training_triplets[tri_str] >= 11 and training_triplets[tri_str] <= n_shots):
                                        ind_zs.append(rel_ind)
                            elif tri_str not in training_triplets:
                                assert n_shots == -1, n_shots
                                ind_zs.append(rel_ind)

                        ind_zs = np.array(ind_zs)

                    if filter_zeroshots:
                        if len(ind_zs) > 0:
                            rels = rels[ind_zs]
                        else:
                            rels = np.zeros((0, 3), dtype=np.int32)

        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if training_triplets and filter_zeroshots:
            assert len(rels) == len(ind_zs), (len(rels), len(ind_zs))

        if training_triplets and filter_empty_rels and len(ind_zs) == 0:
            split_mask[image_index[i]] = 0
            continue

        if filter_non_overlap:
            assert mode == 'train'
            inters = bbox_overlaps(boxes_i, boxes_i)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, relationships


def load_info(info_file):
    """
    Loads the file containing the visual genome label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
    """
    info = json.load(open(info_file, 'r'))
    info['label_to_idx']['__background__'] = 0
    info['predicate_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return ind_to_classes, ind_to_predicates


def vg_collate(data, num_gpus=1, is_train=False, mode='det', torch_detector=False, is_cuda=True):
    assert mode in ('det', 'rel')
    blob = Blob(mode=mode, is_train=is_train, num_gpus=num_gpus,
                batch_size_per_gpu=len(data) // num_gpus, torch_detector=torch_detector, is_cuda=is_cuda)
    for d in data:
        blob.append(d)
    blob.reduce()
    return blob


class VGDataLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def splits(cls,
               data_dir,
               backbone='vgg16',
               batch_size=1,
               num_workers=0,
               num_gpus=1,
               is_cuda=True,
               **kwargs):

        mode = 'rel'

        train_data, val_data_dict = VG.splits(data_dir=data_dir,
                                              torch_detector=backbone != 'vgg16_old',
                                              **kwargs)

        train_load = cls(
            dataset=train_data,
            batch_size=batch_size * num_gpus,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=True,
                                            torch_detector=train_data.torch_detector, is_cuda=is_cuda),
            drop_last=True
        )

        val_loaders = {}
        for name, val_data in val_data_dict.items():
            if val_data is None:
                val_load = None
            else:
                val_load = cls(
                    dataset=val_data,
                    batch_size=batch_size * num_gpus if mode=='det' else num_gpus,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=False,
                                                    torch_detector=val_data.torch_detector, is_cuda=is_cuda),
                    drop_last=False
                )
            val_loaders[name] = val_load

        return train_load, val_loaders



def filter_dups(gt_rels, random_edge=True):
    all_rel_sets = defaultdict(list)
    for (o0, o1, r) in gt_rels:
        all_rel_sets[(o0, o1)].append(r)

    gt_rels = [(k[0], k[1], np.random.choice(v) if random_edge else v[0]) for k, v in all_rel_sets.items()]

    return np.array(gt_rels)


