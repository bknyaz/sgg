import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from sgg_models.rel_model_base import RelModelBase
from lib.rel_assignments import rel_assignments
from lib.surgery import filter_dets
from config import NO_GRAD


class RelModelStanford(RelModelBase):

    def __init__(self,
                 train_data,
                 hidden_dim=512,
                 mp_iter=3,
                 **kwargs):
        """
        Message Passing Model from Scene Graph Generation by Iterative Message Passing (https://arxiv.org/abs/1701.02426)
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        """
        super(RelModelStanford, self).__init__(train_data, **kwargs)

        print(self.mode, self.backbone, self.RELS_PER_IMG, self.use_bias, self.test_bias, self.require_overlap)

        self.hidden_dim = hidden_dim

        self.rel_fc = nn.Linear(hidden_dim, self.num_rels)
        self.obj_fc = nn.Linear(hidden_dim, self.num_classes)

        self.obj_unary = nn.Linear(self.obj_dim, hidden_dim)
        self.edge_unary = nn.Linear(self.obj_dim, hidden_dim)


        self.edge_gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.node_gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        self.mp_iter = mp_iter

        self.sub_vert_w_fc = nn.Sequential(nn.Linear(hidden_dim*2, 1), nn.Sigmoid())
        self.obj_vert_w_fc = nn.Sequential(nn.Linear(hidden_dim*2, 1), nn.Sigmoid())
        self.out_edge_w_fc = nn.Sequential(nn.Linear(hidden_dim*2, 1), nn.Sigmoid())

        self.in_edge_w_fc = nn.Sequential(nn.Linear(hidden_dim*2, 1), nn.Sigmoid())


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

        hx_rel = Variable(rel_rep.data.new(rel_rep.size(0), self.hidden_dim).zero_(), requires_grad=False)
        hx_obj = Variable(obj_rep.data.new(obj_rep.size(0), self.hidden_dim).zero_(), requires_grad=False)

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

        # combine edge visual and box features
        edge_feat = self.union_boxes(edge_feat.view(edge_feat.shape[0], -1, self.pool_sz, self.pool_sz),
                                     rois, rel_inds[:, 1:], im_sizes)

        node_feat = self.obj_unary(self.roi_fmap_obj(node_feat.view(node_feat.shape[0], -1)))
        edge_feat = F.relu(self.edge_unary(self.roi_fmap(edge_feat)))
        node_feat, edge_features = self.message_pass(edge_feat, node_feat, rel_inds[:, 1:3])

        return self.obj_fc(node_feat), self.rel_fc(edge_features)


    def forward(self, batch):
        """
        Forward pass for detection

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param gt_rels: [num_gt_rels, 4] gt relationships where each one is (img_id, subj_id, obj_id, class)

        """

        assert len(batch) == 1, ('single GPU is only supported in this code', len(batch))

        x, gt_boxes, gt_classes, gt_rels = batch[0][0], batch[0][3], batch[0][4], batch[0][5]

        with NO_GRAD():  # do not update anything in the detector
            if self.backbone == 'vgg16_old':
                raise NotImplementedError('%s is not supported any more' % self.backbone)
            else:
                result = self.faster_rcnn(x, gt_boxes, gt_classes, gt_rels)

        result.fmap = result.fmap.detach()  # do not update the detector

        im_inds = result.im_inds
        boxes = result.rm_box_priors

        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                0, filter_non_overlap=True, num_sample_per_gt=1)
        elif not hasattr(result, 'rel_labels'):
            result.rel_labels = None

        rel_inds = self.get_rel_inds(result.rel_labels if self.training else None, im_inds, boxes)
        result.rel_inds = rel_inds
        rois = torch.cat((im_inds[:, None].float(), boxes), 1)

        result.node_feat, result.edge_feat = self.node_edge_features(result.fmap,
                                                                     rois,
                                                                     rel_inds[:, 1:],
                                                                     im_sizes=result.im_sizes)

        result.rm_obj_dists, result.rel_dists = self.predict(result.node_feat,
                                                             result.edge_feat,
                                                             rel_inds,
                                                             rois=rois,
                                                             im_sizes=result.im_sizes)

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
            result.rois = rois
            return result

        if self.mode == 'predcls':
            result.obj_scores = result.rm_obj_dists.data.new(gt_classes.shape[0]).fill_(1)
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
        if self.backbone != 'vgg16_old':
            bboxes = result.rm_box_priors_org
        else:
            bboxes = result.rm_box_priors

        rel_rep = F.softmax(result.rel_dists, dim=1)

        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:], rel_rep)
