import torch
from torch.nn.functional import cross_entropy as CE


def edge_losses(rel_dists, rel_labels, loss_type='dnorm', idx_fg=None, idx_bg=None,
                return_idx=False, loss_weights=(1,1,1), sfx=''):
    '''
    Predicate classification loss. Based on [1].

    [1] B. Knyazev, H. de Vries, C. Cangea, G.W. Taylor, A. Courville, E. Belilovsky.
    Graph Density-Aware Losses for Novel Compositions in Scene Graph Generation. BMVC 2020.
    https://arxiv.org/abs/2005.08230

    :param rel_dists:
    :param rel_labels:
    :param loss_type:
    :param idx_fg:
    :param idx_bg:
    :param return_idx:
    :param abg:
    :param sfx:
    :return:
    '''
    losses = {}

    loss = CE(rel_dists, rel_labels, reduction='none')  # per edge loss

    if idx_fg is None:
        idx_fg = torch.nonzero(rel_labels > 0).data.view(-1)

    if idx_bg is None:
        idx_bg = torch.nonzero(rel_labels == 0).data.view(-1)

    M_FG, M_BG, M = len(idx_fg), len(idx_bg), len(rel_dists)
    assert M == len(rel_labels), (M, len(rel_labels))

    alpha, beta, gamma = loss_weights

    if loss_type == 'baseline':

        assert alpha == beta == 1, ('wrong loss is used, use dnorm or dnorm-fgbg', alpha, beta)
        loss = gamma * (loss / M)  # weight all edges by the same value (divide by M to compute average below)
        losses['rel_loss' + sfx] = loss.sum()  # loss is averaged over all FG and BG edges

    elif loss_type in ['dnorm', 'dnorm-fgbg']:

        edge_weights = torch.ones(M).to(rel_dists)

        # Weight for foreground (annotated) edges
        if M_FG > 0:
            edge_weights[idx_fg] = float(alpha) / M_FG   # weight for FG edges (alpha/M_FG instead of 1/M as in the baseline)

        # Weight for background (not annotated) edges
        if loss_type == 'dnorm':
            # conf.alpha = conf.beta = 1 in our hyperparameter-free loss
            if M_BG > 0 and M_FG > 0:
                edge_weights[idx_bg] = float(beta) / M_FG   # weight for BG edges (beta/M_FG instead of 1/M as in the baseline)
        else:
            if M_BG > 0:
                edge_weights[idx_bg] = float(beta) / M_BG   # weight for BG edges (beta/M_BG instead of 1/M as in the baseline)

        loss = gamma * loss * torch.autograd.Variable(edge_weights)
        losses['rel_loss' + sfx] = loss.sum()
    else:
        raise NotImplementedError(loss_type)

    if return_idx:
        return losses, idx_fg, idx_bg
    else:
        return losses


def node_losses(rm_obj_dists, rm_obj_labels, sfx=''):
    return { 'obj_loss' + sfx: CE(rm_obj_dists, rm_obj_labels) }
