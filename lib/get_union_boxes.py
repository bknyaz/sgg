import numpy as np
import torch
from torch.nn import functional as F
from lib.pytorch_misc import enumerate_by_image
from torch.nn.modules.module import Module
from torch import nn
from config import BATCHNORM_MOMENTUM, TORCH12
# from lib.draw_rectangles.draw_rectangles import draw_union_boxes

if TORCH12:
    from torchvision.ops import roi_align
else:
    # pytorch 0.3
    from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction  # need to compile fpn code from Neural Motifs


class UnionBoxesAndFeats(Module):
    def __init__(self, edge_model='motifs', pooling_size=7, stride=16, dim=256, concat=False, use_feats=True):
        """
        :param pooling_size: Pool the union boxes to this dimension
        :param stride: pixel spacing in the entire image
        :param dim: Dimension of the feats
        :param concat: Whether to concat (yes) or add (False) the representations
        """
        super(UnionBoxesAndFeats, self).__init__()

        self.edge_model = edge_model
        if self.edge_model == 'motifs':
            try:
                from lib.draw_rectangles.draw_rectangles import draw_union_boxes
            except:
                print('Error importing draw_rectangles, which means that most likely this module was not built properly (see README.md)')
                raise
            self.draw_union_boxes = draw_union_boxes
        elif self.edge_model == 'raw_boxes':
            self.draw_union_boxes = draw_union_boxes_grid
        else:
            raise NotImplementedError(self.edge_model)

        conv_layer = lambda n_in, n_out, ks, stide, pad, bias: nn.Conv2d(n_in, n_out,
                                                                         kernel_size=ks,
                                                                         stride=stride,
                                                                         padding=pad, bias=bias)

        self.pooling_size = pooling_size
        self.stride = stride

        self.dim = dim
        self.use_feats = use_feats

        self.conv = nn.Sequential(
            conv_layer(2, dim //2, 7, 2, 3, True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim//2, momentum=BATCHNORM_MOMENTUM),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_layer(dim // 2, dim, 3, 1, 1, True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim, momentum=BATCHNORM_MOMENTUM)  # remove batch norm here to make features relu'ed
        )
        self.concat = concat


    def forward(self, union_pools, rois, union_inds, im_sizes):

        if self.edge_model == 'motifs':
            pair_rois = torch.cat((rois[:, 1:][union_inds[:, 0]], rois[:, 1:][union_inds[:, 1]]), 1).data.cpu().numpy()
            rects = torch.from_numpy(self.draw_union_boxes(pair_rois, self.pooling_size * 4 - 1) - 0.5).to(union_pools)
        elif self.edge_model == 'raw_boxes':
            boxes = rois[:, 1:].clone()
            # scale boxes to the range [0,1]
            scale = boxes.new(boxes.shape).fill_(0).float()
            for i, s, e in enumerate_by_image(rois[:, 0].long().data):
                h, w = im_sizes[i][:2]
                scale[s:e, 0] = w
                scale[s:e, 1] = h
                scale[s:e, 2] = w
                scale[s:e, 3] = h
            boxes = boxes / scale

            try:
                rects = self.draw_union_boxes(boxes, union_inds, self.pooling_size * 4 - 1) - 0.5
            except Exception as e:
                # there was a problem with bboxes being larger than images at test time, had to clip them
                print(rois, boxes, im_sizes, scale)
                raise

            # to debug:
            # print('rects my', rects.shape, rects.min(), rects.max())
            # np.save('rects.npy', rects.data.cpu().numpy())
            # pair_rois = torch.cat((rois[:, 1:][union_inds[:, 0]], rois[:, 1:][union_inds[:, 1]]), 1).data.cpu().numpy()
            # rects2 = torch.from_numpy(draw_union_boxes(pair_rois, self.pooling_size * 4 - 1) - 0.5).to(union_pools)
            # print('rects2', rects2.shape, rects2.min(), rects2.max())
            # np.save('rects2.npy', rects2.data.cpu().numpy())
            # print(union_inds)
            # raise ValueError('saved')
        else:
            raise NotImplementedError(self.edge_model)

        if self.concat:
            return torch.cat((union_pools, self.conv(rects)), 1)
        return union_pools + self.conv(rects)
 


def draw_union_boxes_grid(boxes, union_inds, sz):
    """

    :param boxes: in range [0,1]
    :param union_inds:
    :param sz:
    :return:
    """
    assert boxes.max() <= 1.01, boxes.max()
    boxes_grid = F.grid_sample(boxes.new(len(boxes), 1, sz, sz).fill_(1), _boxes_to_grid(boxes, sz, sz))
    out = boxes_grid[union_inds.reshape(-1)].reshape(len(union_inds), 2, sz, sz)
    return out


def _boxes_to_grid(boxes, H, W):
    # Copied from https://github.com/google/sg2im/blob/master/sg2im/layout.py#L94

    """
    Input:
    - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1]
      format in the [0, 1] coordinate space
    - H, W: Scalars giving size of output

    Returns:
    - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
    """
    O = boxes.size(0)

    boxes = boxes.view(O, 4, 1, 1)

    # All these are (O, 1, 1)
    x0, y0 = boxes[:, 0], boxes[:, 1]
    x1, y1 = boxes[:, 2], boxes[:, 3]
    ww = x1 - x0
    hh = y1 - y0

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww  # (O, 1, W)
    Y = (Y - y0) / hh  # (O, H, 1)

    # Stack does not broadcast its arguments so we need to expand explicitly
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    grid = grid.mul(2).sub(1)

    return grid


def get_node_edge_features(fmap, rois, union_rois=None, union_inds=None, pooling_size=14, stride=16):
    """
    :param fmap: (batch_size, d, IM_SIZE/stride, IM_SIZE/stride)
    :param rois: (num_rois, 5) with [im_ind, x1, y1, x2, y2]
    :param union_inds: (num_urois, 2) with [roi_ind1, roi_ind2]
    :param pooling_size: we'll resize to this
    :param stride:
    :return:
    """
    if union_rois is None:
        assert union_inds.size(1) == 2, union_inds.shape
        union_rois = torch.cat((
            rois[:, 0][union_inds[:, 0]][:, None],
            torch.min(rois[:, 1:3][union_inds[:, 0]], rois[:, 1:3][union_inds[:, 1]]),
            torch.max(rois[:, 3:5][union_inds[:, 0]], rois[:, 3:5][union_inds[:, 1]]),
        ),1)

    if TORCH12:
        node_features = roi_align(fmap, rois, output_size=(pooling_size, pooling_size),
                                  spatial_scale=1. / stride, sampling_ratio=-1)
        edge_features = roi_align(fmap, union_rois, output_size=(pooling_size, pooling_size),
                                  spatial_scale=1. / stride, sampling_ratio=-1)
    else:
        node_features = RoIAlignFunction(pooling_size, pooling_size, spatial_scale=1. / stride)(fmap, rois)
        edge_features = RoIAlignFunction(pooling_size, pooling_size, spatial_scale=1. / stride)(fmap, union_rois)

    return node_features, edge_features