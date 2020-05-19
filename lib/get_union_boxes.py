import torch
from torch.nn import functional as F
from lib.pytorch_misc import enumerate_by_image
from torch.nn.modules.module import Module
from torch import nn
from config import BATCHNORM_MOMENTUM


class UnionBoxesAndFeats(Module):
    def __init__(self, pooling_size=7, stride=16, dim=256, concat=False, use_feats=True, SN=False):
        """
        :param pooling_size: Pool the union boxes to this dimension
        :param stride: pixel spacing in the entire image
        :param dim: Dimension of the feats
        :param concat: Whether to concat (yes) or add (False) the representations
        """
        super(UnionBoxesAndFeats, self).__init__()

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

        boxes = rois[:, 1:].clone()
        multiplier = boxes.new(boxes.shape).fill_(0)
        for i, s, e in enumerate_by_image(rois[:, 0].long().data):
            h, w = im_sizes[i][:2]
            multiplier[s:e, 0] = w
            multiplier[s:e, 1] = h
            multiplier[s:e, 2] = w
            multiplier[s:e, 3] = h
        boxes = boxes / multiplier

        rects = draw_union_boxes_my(boxes, union_inds, self.pooling_size * 4 - 1) - 0.5
        if self.concat:
            return torch.cat((union_pools, self.conv(rects)), 1)
        return union_pools + self.conv(rects)
 


def draw_union_boxes_my(boxes, union_inds, sz):
    """

    :param boxes: in range [0,1]
    :param union_inds:
    :param sz:
    :return:
    """
    assert boxes.max() <= 1.1, boxes.max()
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
