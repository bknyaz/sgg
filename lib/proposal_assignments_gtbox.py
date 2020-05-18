from lib.pytorch_misc import enumerate_by_image, gather_nd, random_choose
import torch
from lib.pytorch_misc import diagonal_inds, to_variable
from config import REL_FG_FRACTION


@to_variable
def proposal_assignments_gtbox(rois, gt_boxes, gt_classes, gt_rels, image_offset, RELS_PER_IMG, fg_thresh=0.5, sample_factor=-1):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    :param rpn_rois: [img_ind, x1, y1, x2, y2]
    :param gt_boxes:   [num_boxes, 4] array of x0, y0, x1, y1]. Not needed it seems
    :param gt_classes: [num_boxes, 2] array of [img_ind, class]
        Note, the img_inds here start at image_offset
    :param gt_rels     [num_boxes, 4] array of [img_ind, box_0, box_1, rel type].
        Note, the img_inds here start at image_offset
    :param Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    :return:
        rois: [num_rois, 5]
        labels: [num_rois] array of labels
        bbox_targets [num_rois, 4] array of targets for the labels.
        rel_labels: [num_rels, 4] (img ind, box0 ind, box1ind, rel type)
    """
    im_inds = rois[:,0].long()

    num_im = im_inds[-1] + 1

    # Offset the image indices in fg_rels to refer to absolute indices (not just within img i)
    fg_rels = gt_rels.clone()
    fg_rels[:,0] -= image_offset
    offset = {}
    for i, s, e in enumerate_by_image(im_inds):
        offset[i] = s
    for i, s, e in enumerate_by_image(fg_rels[:, 0]):
        fg_rels[s:e, 1:3] += offset[i]

    # Try ALL things, not just intersections.
    is_cand = (im_inds[:, None] == im_inds[None])
    is_cand.view(-1)[diagonal_inds(is_cand)] = 0

    # NOW WE HAVE TO EXCLUDE THE FGs.
    is_cand.view(-1)[fg_rels[:,1]*im_inds.size(0) + fg_rels[:,2]] = 0
    is_bgcand = is_cand.nonzero()
    # TODO: make this sample on a per image case
    # If too many then sample
    num_fg = min(fg_rels.size(0), int(RELS_PER_IMG * REL_FG_FRACTION * num_im))
    if num_fg < fg_rels.size(0):
        fg_rels = random_choose(fg_rels, num_fg)

    # If too many then sample
    is_train = num_im > 1  # assume num_im = 1 at test time (except for the det mode, which we don't use for now)
    sample_bg = is_train and sample_factor > -1

    num_bg = min(is_bgcand.size(0) if is_bgcand.dim() > 0 else 0,
                 int(num_fg * sample_factor) if sample_bg else (int(RELS_PER_IMG * num_im) - num_fg) )  # sample num_fg at training time

    if num_bg > 0:
        bg_rels = torch.cat((
            im_inds[is_bgcand[:, 0]][:, None],
            is_bgcand,
            (is_bgcand[:, 0, None] < -10).long(),
        ), 1)

        if num_bg < is_bgcand.size(0):
            bg_rels = random_choose(bg_rels, num_bg)  # at test time will correspond to the baseline approach

        rel_labels = torch.cat((fg_rels, bg_rels), 0)
    else:
        rel_labels = fg_rels


    # last sort by rel.
    _, perm = torch.sort(rel_labels[:, 0]*(gt_boxes.size(0)**2) +
                         rel_labels[:,1]*gt_boxes.size(0) + rel_labels[:,2])

    rel_labels = rel_labels[perm].contiguous()

    labels = gt_classes[:,1].contiguous()
    return rois, labels, rel_labels
