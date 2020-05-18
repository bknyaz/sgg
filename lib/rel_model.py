"""
Base class for relationship models
"""

import torch.nn as nn
import torch.nn.parallel
import torchvision
import copy

from lib.get_union_boxes import UnionBoxesAndFeats
from lib.pytorch_misc import diagonal_inds, bbox_overlaps, gather_res
from lib.sparse_targets import FrequencyBias
from config import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MODES = ('sgdet', 'sgcls', 'predcls')

class RelModelBase(nn.Module):
    """
    RELATIONSHIPS
    """
    def __init__(self, train_data, mode='sgdet', num_gpus=1,
                 require_overlap_det=True,
                 use_bias=False, test_bias=False,
                 detector_model='baseline', RELS_PER_IMG=1024):

        """
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param require_overlap_det: Whether two objects must intersect
        """
        super(RelModelBase, self).__init__()
        self.classes = train_data.ind_to_classes
        self.rel_classes = train_data.ind_to_predicates
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode
        self.detector_model = detector_model
        self.RELS_PER_IMG = RELS_PER_IMG
        self.pooling_size = 7
        self.stride = 16
        self.obj_dim = 4096

        self.use_bias = use_bias
        self.test_bias = test_bias

        self.require_overlap = require_overlap_det and self.mode == 'sgdet'

        if self.detector_model == 'mrcnn':
            print('\nLoading COCO pretrained model maskrcnn_resnet50_fpn...\n')
            # See https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
            self.detector = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                               box_detections_per_img=50,
                                                                               box_score_thresh=0.2)
            in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.classes))
            self.detector.roi_heads.mask_predictor = None
            

        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=self.stride,
                                              dim=256 if self.detector_model == 'mrcnn' else 512)

        if self.detector_model == 'mrcnn':
            layers = list(self.detector.roi_heads.children())[:2]
            self.roi_fmap_obj = copy.deepcopy(layers[1])
            self.roi_fmap = copy.deepcopy(layers[1])
            self.multiscale_roi_pool = copy.deepcopy(layers[0])
        else:
            raise NotImplementedError(self.detector_model)

        if self.use_bias:
            self.freq_bias = FrequencyBias(train_data)


    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        # Get the relationship candidates
        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)

                # if there are fewer then 100 things then we might as well add some?
                amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)

        return rel_inds


    def predict(self, node_feat, edge_feat, rel_inds, rois, im_sizes):
        raise NotImplementedError('predict')


    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, *arg):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels

            if test:
            prob dists, boxes, img inds, maxscores, classes

        """

        raise NotImplementedError('forward')


    def forward_parallel(self, batch):
        # define as a function here so that wandb can watch gradients
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        else:
            raise NotImplementedError('need to make sure it is correct')

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs

