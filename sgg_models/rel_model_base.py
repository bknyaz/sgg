"""
Base class for relationship models
"""

import torch.nn as nn
import torch.nn.parallel
import copy
from collections import OrderedDict

from lib.proposal_assignments_gtbox import proposal_assignments_gtbox
from lib.get_union_boxes import UnionBoxesAndFeats, get_node_edge_features
from lib.pytorch_misc import diagonal_inds, bbox_overlaps, enumerate_by_image, Result
from lib.sparse_targets import FrequencyBias
from config import IM_SCALE

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


class RelModelBase(nn.Module):
    """
    RELATIONSHIPS
    """

    def __init__(self,
                 train_data,
                 mode='sgcls',
                 require_overlap_det=True,
                 use_bias=False,
                 test_bias=False,
                 backbone='vgg16',
                 RELS_PER_IMG=1024,
                 min_size=None,
                 max_size=None,
                 edge_model='motifs'):

        """
        Base class for an SGG model
        :param mode: (sgcls, predcls, or sgdet)
        :param require_overlap_det: Whether two objects must intersect
        """
        super(RelModelBase, self).__init__()
        self.classes = train_data.ind_to_classes
        self.rel_classes = train_data.ind_to_predicates
        self.mode = mode
        self.backbone = backbone
        self.RELS_PER_IMG = RELS_PER_IMG
        self.pool_sz = 7
        self.stride = 16

        self.use_bias = use_bias
        self.test_bias = test_bias

        self.require_overlap = require_overlap_det and self.mode == 'sgdet'

        if self.backbone == 'resnet50':
            self.obj_dim = 1024
            self.fmap_sz = 21

            if min_size is None:
                min_size = 1333
            if max_size is None:
                max_size = 1333

            print('\nLoading COCO pretrained model maskrcnn_resnet50_fpn...\n')
            # See https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
            self.detector = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                               min_size=min_size, max_size=max_size,
                                                                               box_detections_per_img=50,
                                                                               box_score_thresh=0.2)
            in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.classes))
            self.detector.roi_heads.mask_predictor = None

            layers = list(self.detector.roi_heads.children())[:2]
            self.roi_fmap_obj = copy.deepcopy(layers[1])
            self.roi_fmap = copy.deepcopy(layers[1])
            self.roi_pool = copy.deepcopy(layers[0])

        elif self.backbone == 'vgg16':
            self.obj_dim = 4096
            self.fmap_sz = 38

            if min_size is None:
                min_size = IM_SCALE
            if max_size is None:
                max_size = IM_SCALE

            vgg = load_vgg(use_dropout=False, use_relu=False, use_linear=True, pretrained=False)
            vgg.features.out_channels = 512
            anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                               aspect_ratios=((0.5, 1.0, 2.0),))

            roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                            output_size=self.pool_sz,
                                                            sampling_ratio=2)

            self.detector = FasterRCNN(vgg.features,
                                       min_size=min_size, max_size=max_size,
                                       rpn_anchor_generator=anchor_generator,
                                       box_head=TwoMLPHead(vgg.features.out_channels * self.pool_sz ** 2, self.obj_dim),
                                       box_predictor=FastRCNNPredictor(self.obj_dim, len(train_data.ind_to_classes)),
                                       box_roi_pool=roi_pooler,
                                       box_detections_per_img=50,
                                       box_score_thresh=0.2)

            self.roi_fmap = nn.Sequential(nn.Flatten(), vgg.classifier)
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier
            self.roi_pool = copy.deepcopy(list(self.detector.roi_heads.children())[0])

        else:
            raise NotImplementedError(self.backbone)

        self.edge_dim = self.detector.backbone.out_channels

        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pool_sz, stride=self.stride,
                                              dim=self.edge_dim, edge_model=edge_model)
        if self.use_bias:
            self.freq_bias = FrequencyBias(train_data)


    @property
    def num_classes(self):
        return len(self.classes)


    @property
    def num_rels(self):
        return len(self.rel_classes)


    def predict(self, node_feat, edge_feat, rel_inds, rois, im_sizes):
        raise NotImplementedError('predict')


    def forward(self, batch):
        raise NotImplementedError('forward')


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
                # amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)

        return rel_inds


    def set_box_score_thresh(self, box_score_thresh):
        if self.backbone != 'vgg16_old':
            self.detector.roi_heads.score_thresh = box_score_thresh
        else:
            self.detector.thresh = box_score_thresh


    def faster_rcnn(self, x, gt_boxes, gt_classes, gt_rels):
        targets, x_lst, original_image_sizes = [], [], []
        device = self.rel_fc.weight.get_device() if self.rel_fc.weight.is_cuda else 'cpu'
        for i, s, e in enumerate_by_image(gt_classes[:, 0].long().data):
            targets.append({'boxes': copy.deepcopy(gt_boxes[s:e]), 'labels': gt_classes[s:e, 1].long()})
            x_lst.append(x[i].to(device).squeeze())
            original_image_sizes.append(x[i].shape[-2:])

        images, targets = self.detector.transform(x_lst, targets)
        fmap_multiscale = self.detector.backbone(images.tensors)
        if isinstance(fmap_multiscale, torch.Tensor):
            fmap_multiscale = OrderedDict([('0', fmap_multiscale)])

        if self.mode != 'sgdet':
            rois, obj_labels, rel_labels = self.gt_labels(gt_boxes, gt_classes, gt_rels)
            rm_box_priors, rm_box_priors_org = [], []
            for i, s, e in enumerate_by_image(gt_classes[:, 0].long().data):
                rm_box_priors.append(targets[i]['boxes'])
                rm_box_priors_org.append(gt_boxes[s:e])

            im_inds = rois[:, 0]
            result = Result(
                od_box_targets=None,
                rm_box_targets=None,
                od_obj_labels=obj_labels,
                rm_box_priors=torch.cat(rm_box_priors),
                rm_obj_labels=obj_labels,
                rpn_scores=None,
                rpn_box_deltas=None,
                rel_labels=rel_labels,
                im_inds=im_inds.long(),
            )
            result.rm_box_priors_org = torch.cat(rm_box_priors_org)

        else:
            proposals, _ = self.detector.rpn(images, fmap_multiscale, targets)
            detections, _ = self.detector.roi_heads(fmap_multiscale, proposals, images.image_sizes, targets)
            boxes = copy.deepcopy(detections)
            boxes_all_dict = self.detector.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            rm_box_priors, rm_box_priors_org, im_inds, obj_labels = [], [], [], []
            for i in range(len(proposals)):
                if len(boxes[i]['boxes']) <= 1:
                    raise ValueError(
                        'at least two objects must be detected to build relationships, make sure the detector is properly pretrained',
                        boxes)
                rm_box_priors.append(boxes[i]['boxes'])
                rm_box_priors_org.append(boxes_all_dict[i]['boxes'])
                obj_labels.append(boxes_all_dict[i]['labels'])
                im_inds.append(torch.zeros(len(detections[i]['boxes'])) + i)

            im_inds = torch.cat(im_inds).to(device)
            result = Result(
                rm_obj_labels=torch.cat(obj_labels).view(-1),
                rm_box_priors=torch.cat(rm_box_priors),
                rel_labels=None,
                im_inds=im_inds.long()
            )
            result.rm_box_priors_org = torch.cat(rm_box_priors_org)

            if len(result.rm_box_priors) <= 1:
                raise ValueError('at least two objects must be detected to build relationships')

        result.im_sizes_org = original_image_sizes
        result.im_sizes = images.image_sizes
        result.fmap = fmap_multiscale[list(fmap_multiscale.keys())[-1]]  # last scale for global feature maps
        result.rois = torch.cat((im_inds.float()[:, None], result.rm_box_priors), 1)

        return result


    def node_edge_features(self, fmap, rois, union_inds, im_sizes):

        assert union_inds.shape[1] == 2, union_inds.shape
        union_rois = torch.cat((rois[:, 0][union_inds[:, 0]][:, None],
                                torch.min(rois[:, 1:3][union_inds[:, 0]], rois[:, 1:3][union_inds[:, 1]]),
                                torch.max(rois[:, 3:5][union_inds[:, 0]], rois[:, 3:5][union_inds[:, 1]])), 1)

        if self.backbone == 'vgg16_old':
            return get_node_edge_features(fmap, rois, union_rois=union_rois,
                                          pooling_size=self.pool_sz, stride=self.stride)
        else:
            if isinstance(fmap, torch.Tensor):
                fmap = OrderedDict([('0', fmap)])
            node_feat = self.roi_pool(fmap, convert_roi_to_list(rois), im_sizes)  # images.image_sizes
            edge_feat = self.roi_pool(fmap, convert_roi_to_list(union_rois), im_sizes)
            return node_feat, edge_feat


    def get_scaled_boxes(self, boxes, im_inds, im_sizes):
        if self.backbone == 'vgg16_old':
            boxes_scaled = boxes / IM_SCALE
        else:
            boxes_scaled = boxes.clone()
            for im_ind, s, e in enumerate_by_image(im_inds.long().data):
                boxes_scaled[s:e, [0, 2]] = boxes_scaled[s:e, [0, 2]] / im_sizes[im_ind][1]  # width
                boxes_scaled[s:e, [1, 3]] = boxes_scaled[s:e, [1, 3]] / im_sizes[im_ind][0]  # height

        assert boxes_scaled.max() <= 1 + 1e-3, (boxes_scaled.max(), boxes.max(), im_sizes)

        return boxes_scaled


    def gt_labels(self, gt_boxes, gt_classes, gt_rels=None, sample_factor=-1):
        """
        Gets GT boxes!
        :param fmap:
        :param im_sizes:
        :param image_offset:
        :param gt_boxes:
        :param gt_classes:
        :param gt_rels:
        :param train_anchor_inds:
        :return:
        """
        assert gt_boxes is not None
        im_inds = gt_classes[:, 0]
        rois = torch.cat((im_inds.float()[:, None], gt_boxes), 1)
        if gt_rels is not None and self.training:
            rois, obj_labels, rel_labels = proposal_assignments_gtbox(
                rois.data, gt_boxes.data, gt_classes.data, gt_rels.data, 0, self.RELS_PER_IMG,
                sample_factor=sample_factor)
        else:
            obj_labels = gt_classes[:, 1]
            rel_labels = None

        return rois, obj_labels, rel_labels


def convert_roi_to_list(rois):
    rois_lst = []
    for im_ind, s, e in enumerate_by_image(rois[:, 0].long().data):
        rois_lst.append(rois[s:e, 1:])
    return rois_lst


def load_vgg(use_dropout=True, use_relu=True, use_linear=True, pretrained=True):
    model = torchvision.models.vgg16(pretrained=pretrained)
    del model.features._modules['30']  # Get rid of the maxpool
    del model.classifier._modules['6']  # Get rid of class layer
    if not use_dropout:
        # del model.classifier._modules['2']
        del model.classifier._modules['5']  # Get rid of dropout
        if not use_relu:
            del model.classifier._modules['4']  # Get rid of relu activation
            if not use_linear:
                del model.classifier._modules['3']  # Get rid of linear layer
    return model
