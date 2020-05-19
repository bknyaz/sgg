
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from lib.rel_assignments import rel_assignments
from lib.rel_model import RelModelBase
from lib.proposal_assignments_gtbox import proposal_assignments_gtbox
import copy
from collections import OrderedDict
from lib.pytorch_misc import enumerate_by_image, Result
from lib.surgery import filter_dets

SIZE=512

class RelModelStanford(RelModelBase):
    """
    RELATIONSHIPS
    """

    def __init__(self, train_data, mode='sgdet', num_gpus=1,
                 require_overlap_det=True,
                 use_bias=False,
                 test_bias=False,
                 detector_model='mrcnn',
                 RELS_PER_IMG=1024, mp_iter=3, **kwargs):
        """
        Message Passing Model from Scene Graph Generation by Iterative Message Passing (https://arxiv.org/abs/1701.02426)
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param num_gpus: how many GPUS 2 use
        """
        super(RelModelStanford, self).__init__(train_data, mode=mode, num_gpus=num_gpus,
                                               require_overlap_det=require_overlap_det,
                                               use_bias=use_bias,
                                               test_bias=test_bias,
                                               detector_model=detector_model,
                                               RELS_PER_IMG=RELS_PER_IMG, **kwargs)

        self.rel_fc = nn.Linear(SIZE, self.num_rels)
        self.obj_fc = nn.Linear(SIZE, self.num_classes)

        self.obj_unary = nn.Linear(self.obj_dim if self.detector_model == 'baseline' else 1024, SIZE)
        self.edge_unary = nn.Linear(self.obj_dim if self.detector_model == 'baseline' else 1024, SIZE)


        self.edge_gru = nn.GRUCell(input_size=SIZE, hidden_size=SIZE)
        self.node_gru = nn.GRUCell(input_size=SIZE, hidden_size=SIZE)

        self.mp_iter = mp_iter

        self.sub_vert_w_fc = nn.Sequential(nn.Linear(SIZE*2, 1), nn.Sigmoid())
        self.obj_vert_w_fc = nn.Sequential(nn.Linear(SIZE*2, 1), nn.Sigmoid())
        self.out_edge_w_fc = nn.Sequential(nn.Linear(SIZE*2, 1), nn.Sigmoid())

        self.in_edge_w_fc = nn.Sequential(nn.Linear(SIZE*2, 1), nn.Sigmoid())


    def message_pass(self, rel_rep, obj_rep, rel_inds):
        """

        :param rel_rep: [num_rel, fc]
        :param obj_rep: [num_obj, fc]
        :param rel_inds: [num_rel, 2] of the valid relationships
        :return: object prediction [num_obj, 151], bbox_prediction [num_obj, 151*4] 
                and rel prediction [num_rel, 51]
        """
        # [num_obj, num_rel] with binary!
        numer = torch.arange(0, rel_inds.size(0), device=rel_inds.get_device() if rel_inds.is_cuda else 'cpu').long()

        objs_to_outrels = rel_rep.data.new(obj_rep.size(0), rel_rep.size(0)).zero_()
        objs_to_outrels.view(-1)[rel_inds[:, 0] * rel_rep.size(0) + numer] = 1
        objs_to_outrels = Variable(objs_to_outrels)

        objs_to_inrels = rel_rep.data.new(obj_rep.size(0), rel_rep.size(0)).zero_()
        objs_to_inrels.view(-1)[rel_inds[:, 1] * rel_rep.size(0) + numer] = 1
        objs_to_inrels = Variable(objs_to_inrels)

        hx_rel = Variable(rel_rep.data.new(rel_rep.size(0), SIZE).zero_(), requires_grad=False)
        hx_obj = Variable(obj_rep.data.new(obj_rep.size(0), SIZE).zero_(), requires_grad=False)

        vert_factor = [self.node_gru(obj_rep, hx_obj)]
        edge_factor = [self.edge_gru(rel_rep, hx_rel)]

        for i in range(self.mp_iter):
            # compute edge context
            sub_vert = vert_factor[i][rel_inds[:, 0]]
            obj_vert = vert_factor[i][rel_inds[:, 1]]
            weighted_sub = self.sub_vert_w_fc(
                torch.cat((sub_vert, edge_factor[i]), 1)) * sub_vert
            weighted_obj = self.obj_vert_w_fc(
                torch.cat((obj_vert, edge_factor[i]), 1)) * obj_vert

            edge_factor.append(self.edge_gru(weighted_sub + weighted_obj, edge_factor[i]))

            # Compute vertex context
            pre_out = self.out_edge_w_fc(torch.cat((sub_vert, edge_factor[i]), 1)) * \
                      edge_factor[i]
            pre_in = self.in_edge_w_fc(torch.cat((obj_vert, edge_factor[i]), 1)) * edge_factor[
                i]

            vert_ctx = objs_to_outrels @ pre_out + objs_to_inrels @ pre_in
            vert_factor.append(self.node_gru(vert_ctx, vert_factor[i]))

        return vert_factor[-1], edge_factor[-1]


    def predict(self, node_feat, edge_feat, rel_inds, rois, im_sizes):

        edge_feat = self.union_boxes(edge_feat.view(edge_feat.shape[0], -1,
                                                    self.pooling_size,
                                                    self.pooling_size), rois,
                                     rel_inds[:,1:], im_sizes)

        node_feat = self.roi_fmap_obj(node_feat)
        edge_feat = self.roi_fmap(edge_feat.contiguous())

        node_feat = self.obj_unary(node_feat)
        edge_feat = F.relu(self.edge_unary(edge_feat))
        obj_features, edge_features = self.message_pass(edge_feat, node_feat, rel_inds[:, 1:3])

        return self.obj_fc(obj_features), self.rel_fc(edge_features)


    def set_box_score_thresh(self, box_score_thresh):
        if self.detector_model == 'mrcnn':
            self.detector.roi_heads.score_thresh = box_score_thresh


    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, *args):
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

        with torch.no_grad():  # do not update anything in the detector

            targets, x_lst, original_image_sizes = [], [], []
            device = self.rel_fc.weight.get_device() if self.rel_fc.weight.is_cuda else 'cpu'
            gt_boxes = gt_boxes.to(device)
            gt_classes = gt_classes.to(device)
            gt_rels = gt_rels.to(device)
            for i, s, e in enumerate_by_image(gt_classes[:, 0].long().data):
                targets.append({ 'boxes': copy.deepcopy(gt_boxes[s:e]), 'labels': gt_classes[s:e, 1].long().to(device) })
                x_lst.append(x[i].to(device).squeeze())
                original_image_sizes.append(x[i].shape[-2:])

            images, targets = self.detector.transform(x_lst, targets)
            fmap_multiscale = self.detector.backbone(images.tensors)
            if self.mode != 'sgdet':
                rois, obj_labels, bbox_targets, rpn_scores, rpn_box_deltas, rel_labels = \
                    self.gt_boxes(None, im_sizes, image_offset, self.RELS_PER_IMG, gt_boxes,
                                   gt_classes, gt_rels, None, proposals=None,
                                   sample_factor=-1)
                rm_box_priors, rm_box_priors_org = [], []
                for i, s, e in enumerate_by_image(gt_classes[:, 0].long().data):
                    rm_box_priors.append(targets[i]['boxes'])
                    rm_box_priors_org.append(gt_boxes[s:e])

                result = Result(
                    od_box_targets=bbox_targets,
                    rm_box_targets=bbox_targets,
                    od_obj_labels=obj_labels,
                    rm_box_priors=torch.cat(rm_box_priors),
                    rm_obj_labels=obj_labels,
                    rpn_scores=rpn_scores,
                    rpn_box_deltas=rpn_box_deltas,
                    rel_labels=rel_labels,
                    im_inds=rois[:, 0].long().contiguous() + image_offset
                )
                result.rm_box_priors_org = torch.cat(rm_box_priors_org)

            else:

                if isinstance(fmap_multiscale, torch.Tensor):
                    fmap_multiscale = OrderedDict([(0, fmap_multiscale)])
                proposals, _ = self.detector.rpn(images, fmap_multiscale, targets)
                detections, _ = self.detector.roi_heads(fmap_multiscale, proposals, images.image_sizes, targets)
                boxes = copy.deepcopy(detections)
                boxes_all_dict = self.detector.transform.postprocess(detections, images.image_sizes, original_image_sizes)
                rm_box_priors, rm_box_priors_org, im_inds, obj_labels = [], [], [], []
                for i in range(len(proposals)):
                    rm_box_priors.append(boxes[i]['boxes'])
                    rm_box_priors_org.append(boxes_all_dict[i]['boxes'])
                    obj_labels.append(boxes_all_dict[i]['labels'])
                    im_inds.append(torch.zeros(len(detections[i]['boxes']),
                                               device=device).float() + i)
                im_inds = torch.cat(im_inds).view(-1, 1)

                result = Result(
                    rm_obj_labels=torch.cat(obj_labels).view(-1),
                    rm_box_priors = torch.cat(rm_box_priors),
                    rel_labels=None,
                    im_inds=im_inds.view(-1).long().contiguous() + image_offset
                )
                result.rm_box_priors_org = torch.cat(rm_box_priors_org)

                if len(result.rm_box_priors) <= 1:
                    raise ValueError('at least two objects must be detected to build relationships')

        if result.is_none():
            return ValueError("heck")

        if self.detector_model == 'baseline':
            if self.slim > 0:
                result.fmap = self.fmap_reduce(result.fmap.detach())
            else:
                result.fmap = result.fmap.detach()

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors

        if not hasattr(result, 'rel_labels'):
            result.rel_labels = None

        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True, num_sample_per_gt=1)

        rel_inds = self.get_rel_inds(result.rel_labels if self.training else None, im_inds, boxes)
        rois = torch.cat((im_inds[:, None].float(), boxes), 1)

        union_rois = torch.cat((
            rois[:, 0][rel_inds[:, 1]][:, None],
            torch.min(rois[:, 1:3][rel_inds[:, 1]], rois[:, 1:3][rel_inds[:, 2]]),
            torch.max(rois[:, 3:5][rel_inds[:, 1]], rois[:, 3:5][rel_inds[:, 2]]),
        ), 1)

        node_feat = self.multiscale_roi_pool(fmap_multiscale, rm_box_priors, images.image_sizes)
        edge_feat = self.multiscale_roi_pool(fmap_multiscale, convert_roi_to_list(union_rois), images.image_sizes)

        result.rm_obj_dists, result.rel_dists = self.predict(node_feat, edge_feat, rel_inds, rois, images.image_sizes)

        if self.use_bias:

            scores_nz = F.softmax(result.rm_obj_dists, dim=1).data
            scores_nz[:, 0] = 0.0
            _, score_ord = scores_nz[:, 1:].sort(dim=1, descending=True)
            result.obj_preds = score_ord[:, 0] + 1

            if self.mode == 'predcls':
                result.obj_preds = gt_classes.data[:, 1]

            freq_pred = self.freq_bias.index_with_labels(torch.stack((
                result.obj_preds[rel_inds[:, 1]],
                result.obj_preds[rel_inds[:, 2]],
            ), 1))
            # tune the weight for freq_bias
            if self.test_bias:
                result.rel_dists = freq_pred
            else:
                result.rel_dists = result.rel_dists + freq_pred

        if self.training:
            return result

        if self.mode == 'predcls':
            result.obj_scores = result.rm_obj_dists.data.new(gt_classes.size(0)).fill_(1)
            result.obj_preds = gt_classes.data[:, 1]
        elif self.mode in ['sgcls', 'sgdet']:
            scores_nz = F.softmax(result.rm_obj_dists, dim=1).data
            scores_nz[:, 0] = 0.0  # does not change actually anything
            result.obj_scores, score_ord = scores_nz[:, 1:].sort(dim=1, descending=True)
            result.obj_preds = score_ord[:, 0] + 1
            result.obj_scores = result.obj_scores[:,0]
        else:
            raise NotImplementedError(self.mode)

        result.obj_preds = Variable(result.obj_preds)
        result.obj_scores = Variable(result.obj_scores)

        # Boxes will get fixed by filter_dets function.
        if self.detector_model == 'mrcnn':
            bboxes = result.rm_box_priors_org
        else:
            bboxes = result.rm_box_priors

        rel_rep = F.softmax(result.rel_dists, dim=1)

        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:], rel_rep)


    def gt_boxes(self, fmap, im_sizes, image_offset, RELS_PER_IMG, gt_boxes=None, gt_classes=None, gt_rels=None,
                 train_anchor_inds=None, proposals=None, sample_factor=-1):
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
        im_inds = gt_classes[:, 0] - image_offset
        rois = torch.cat((im_inds.float()[:, None], gt_boxes), 1)
        if gt_rels is not None and self.training:
            rois, labels, rel_labels = proposal_assignments_gtbox(
                rois.data, gt_boxes.data, gt_classes.data, gt_rels.data, image_offset, RELS_PER_IMG,
                fg_thresh=0.5, sample_factor=sample_factor)
        else:
            labels = gt_classes[:, 1]
            rel_labels = None

        return rois, labels, None, None, None, rel_labels


def convert_roi_to_list(rois):
    rois_lst = []
    for im_ind, s, e in enumerate_by_image(rois[:, 0].long().data):
        rois_lst.append(rois[s:e, 1:])

    return rois_lst

