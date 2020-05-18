"""
Adapted from Danfei Xu. In particular, slow code was removed
"""
import numpy as np
from functools import reduce
from lib.pytorch_misc import intersect_2d, argsort_desc, bbox_overlaps
import math
import pickle
from config import MODES
np.set_printoptions(precision=3)

MAX_RECALL_K = 300


class BasicSceneGraphEvaluator:
    def __init__(self, mode, multiple_preds=False, triplet_counts=None, triplet2str=None, per_triplet=False):
        self.result_dict = {}
        self.mode = mode
        self.result_dict[self.mode + '_recall'] = {20: [], 50: [], 100: [], 200: [], 300: []}

        self.per_triplet = per_triplet
        self.multiple_preds = multiple_preds

        if self.per_triplet:
            self.result_dict[self.mode + '_recall_norm'] = {20: [], 50: [], 100: [], 200: [], 300: []}
            self.result_dict[self.mode + '_rank'] = []
            self.result_dict[self.mode + '_counts'] = []
            self.result_dict[self.mode + '_recall_triplet'] = {5: [], 10: [], 15: [], 20: [], 50: []}
            self.result_dict[self.mode + '_meanrank_triplet'] = []
            self.result_dict[self.mode + '_medianrank_triplet'] = []
            self.result_dict[self.mode + '_medianrankclass_triplet'] = []

            self.result_dict[self.mode + '_recall_triplet_norm'] = {5: [], 10: [], 15: [], 20: [], 50: []}
            self.result_dict[self.mode + '_meanrank_triplet_norm'] = []
            self.result_dict[self.mode + '_medianrank_triplet_norm'] = []
            self.triplet_counts = triplet_counts
            self.triplet2str = triplet2str
            self.triplet_ranks = {}


    @classmethod
    def all_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, **kwargs) for m in MODES}
        return evaluators

    @classmethod
    def vrd_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, multiple_preds=True, **kwargs) for m in ('preddet', 'phrdet')}
        return evaluators

    def evaluate_scene_graph_entry(self, gt_entry, pred_scores, viz_dict=None, iou_thresh=0.5):
        res = self.evaluate_from_dict(gt_entry, pred_scores, self.mode, self.result_dict,
                                  viz_dict=viz_dict, iou_thresh=iou_thresh, multiple_preds=self.multiple_preds)
        # self.print_stats()
        return res

    def save(self, fn):
        np.save(fn, self.result_dict)

    def normalize_counts(self, counts):
        weights = 1. / (counts + 1)  # +1 to avoid zero division, more frequent triplet classes get less weight
        weights /= weights.sum()  # make sure it sums to 1
        return weights

    def print_stats(self, verbose=True):
        output = {}

        items = list(self.result_dict[self.mode + '_recall'].items())
        if verbose:
            print('================' + self.mode + ('(NO GC)' if self.multiple_preds else '(GC)') + ': %d images =================='
                  % len(items[0][1]))
            for k, v in items:
                print('R@%i: %f' % (k, np.mean(v)))
                output['R@%i' % k] = np.mean(v)

        if self.per_triplet:
            ranks = np.array(self.result_dict[self.mode + '_rank']).astype(np.float32).copy()
            counts = np.array(self.result_dict[self.mode + '_counts']).astype(np.float32).copy()
            if verbose:
                print('\nTriplet level evaluation (%d triplets)' % len(ranks))
            weights = self.normalize_counts(counts)
            for k in self.result_dict[self.mode + '_recall_triplet']:
                rec = (ranks < k)
                r = rec.mean()
                rec_norm = (rec.astype(np.float32) * weights)
                r_norm = rec_norm.sum()
                self.result_dict[self.mode + '_recall_triplet'][k] = r
                self.result_dict[self.mode + '_recall_triplet_norm'][k] = r_norm
                if verbose:
                    print('Triplet level R@%i: %.4f (normalized: %.4f)' %
                          (k, r, r_norm))

            m1, m1_std = ranks.mean(), ranks.std()
            m2 = np.median(ranks)
            m1_norm, m1_norm_std = (ranks * weights).sum(), (ranks * weights).std()

            # Weighted median
            triplet_medians, counts = [], []
            for k, v in self.triplet_ranks.items():
                if len(v) > 0:
                    triplet_medians.append(np.median(v))
                    counts.append(self.triplet_counts[k] if k in self.triplet_counts else 0)
            triplet_medians, counts = np.array(triplet_medians), np.array(counts)
            weights = self.normalize_counts(counts)
            m2_class = (triplet_medians).mean()
            m2_class_norm = (triplet_medians * weights).sum()

            self.result_dict[self.mode + '_meanrank_triplet'] = m1
            self.result_dict[self.mode + '_meanrank_triplet_norm'] = m1_norm
            self.result_dict[self.mode + '_medianrank_triplet'] = m2
            self.result_dict[self.mode + '_medianrankclass_triplet'] = m2_class
            self.result_dict[self.mode + '_medianrank_triplet_norm'] = m2_class_norm

            if verbose:
                print('Triplet level mean rank: %.4f (normalized: %.4f)' %
                      (m1, m1_norm))
                print('Triplet level median rank: %.4f (per class: %.4f, normalized per class: %.4f)\n' %
                      (m2, m2_class, m2_class_norm))

        return output


    def evaluate_from_dict(self, gt_entry, pred_entry, mode, result_dict, multiple_preds=False,
                           viz_dict=None, **kwargs):
        """
        Shortcut to doing evaluate_recall from dict
        :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
        :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
        :param mode: 'det' or 'cls'
        :param result_dict:
        :param viz_dict:
        :param kwargs:
        :return:
        """
        gt_rels = gt_entry['gt_relations']
        gt_boxes = gt_entry['gt_boxes'].astype(float)
        gt_classes = gt_entry['gt_classes']

        pred_rel_inds = pred_entry['pred_rel_inds']
        rel_scores = pred_entry['rel_scores']

        if mode == 'predcls':
            pred_boxes = gt_boxes
            pred_classes = gt_classes
            obj_scores = np.ones(gt_classes.shape[0])
        elif mode == 'sgcls':
            pred_boxes = gt_boxes
            pred_classes = pred_entry['pred_classes']
            obj_scores = pred_entry['obj_scores']
        elif mode == 'objcls':
            pred_boxes = gt_boxes
            pred_classes = pred_entry['pred_classes']
            obj_scores = pred_entry['obj_scores']

            # same as sgcls but assume perfect predicate recognition
            pred_rel_inds = gt_rels[:, :2]
            rel_scores = np.zeros((len(gt_rels), rel_scores.shape[1]))
            rel_scores[np.arange(len(gt_rels)), gt_rels[:, 2]] = 1

        elif mode == 'sgdet' or mode == 'phrdet':
            pred_boxes = pred_entry['pred_boxes'].astype(float)
            pred_classes = pred_entry['pred_classes']
            obj_scores = pred_entry['obj_scores']
        elif mode == 'preddet':
            # Only extract the indices that appear in GT
            prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
            if prc.size == 0:
                for k in result_dict[mode + '_recall']:
                    result_dict[mode + '_recall'][k].append(0.0)
                if self.per_triplet:
                    for k in result_dict[mode + '_recall_norm']:
                        result_dict[mode + '_recall_norm'][k].append(0.0)
                return None, None, None
            pred_inds_per_gt = prc.argmax(0)
            pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
            rel_scores = rel_scores[pred_inds_per_gt]

            # Now sort the matching ones
            rel_scores_sorted = argsort_desc(rel_scores[:,1:])
            rel_scores_sorted[:,1] += 1
            rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

            matches = intersect_2d(rel_scores_sorted, gt_rels)
            for k in result_dict[mode + '_recall']:
                rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
                result_dict[mode + '_recall'][k].append(rec_i)
            if self.per_triplet:
                for k in result_dict[mode + '_recall_norm']:
                    rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
                    result_dict[mode + '_recall_norm'][k].append(rec_i)
            return None, None, None
        else:
            raise ValueError('invalid mode')

        if multiple_preds:
            obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
            overall_scores = obj_scores_per_rel[:,None] * rel_scores[:,1:]
            score_inds = argsort_desc(overall_scores)[:MAX_RECALL_K]
            pred_rels = np.column_stack((pred_rel_inds[score_inds[:,0]], score_inds[:,1]+1))
            predicate_scores = rel_scores[score_inds[:,0], score_inds[:,1]+1]
        else:
            pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
            predicate_scores = rel_scores[:,1:].max(1)


        # print('eval', gt_rels.shape, pred_rels.shape, predicate_scores.shape, gt_boxes.shape)
        pred_to_gt, pred_5ples, rel_scores = evaluate_recall(
            gt_rels, gt_boxes, gt_classes,
            pred_rels, pred_boxes, pred_classes,
            predicate_scores, obj_scores, phrdet=mode == 'phrdet',
            **kwargs)

        if self.per_triplet:
            counts = np.zeros(len(gt_rels))
            for rel_i, gt_rel in enumerate(gt_rels):
                o, s, R = gt_rel
                tri_str = '{}_{}_{}'.format(gt_classes[o], R, gt_classes[s])
                if tri_str in self.triplet_counts:
                    counts[rel_i] = self.triplet_counts[tri_str]

            weights = self.normalize_counts(counts)

        for k in result_dict[mode + '_recall']:

            match = reduce(np.union1d, pred_to_gt[:k])
            # print('match', match, type(match))
            match = np.array(match).astype(np.int)

            rec_i = float(len(match)) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)

            if self.per_triplet:
                result_dict[mode + '_recall_norm'][k].append(np.sum(weights[match]))


        if self.per_triplet:
            # TODO: this looks similar to preddet, reuse that code

            score_inds = argsort_desc(overall_scores)
            pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1] + 1))

            # Naive and slow code to get per triplet ranks
            ranks, counts = np.zeros(len(gt_rels)) - 1, np.zeros(len(gt_rels))
            for rel_i, gt_rel in enumerate(gt_rels):
                o, s, R = gt_rel
                tri_str = '{}_{}_{}'.format(gt_classes[o], R, gt_classes[s])
                if tri_str in self.triplet_counts:
                    counts[rel_i] = self.triplet_counts[tri_str]

                # select only pairs with this bounding boxes
                ind = np.where((pred_rels[:, 0] == o) & (pred_rels[:, 1] == s) |
                               (pred_rels[:, 0] == s) & (pred_rels[:, 1] == o))[0]
                pred_to_gt_triplet, _, _ = evaluate_recall(gt_rel.reshape(1, -1), gt_boxes,
                                                                          gt_classes, pred_rels[ind],
                                                                          pred_boxes, pred_classes)

                for r, p in enumerate(pred_to_gt_triplet):
                    if len(p) > 0:
                        assert p == [0], (p, gt_rel, pred_to_gt_triplet)
                        ranks[rel_i] = r
                        break

                if ranks[rel_i] < 0:
                    ranks[rel_i] = MAX_RECALL_K + 1
                # For sgcls not all combinations are present, so take some max rank as the default value

                if tri_str not in self.triplet_ranks:
                    self.triplet_ranks[tri_str] = []
                self.triplet_ranks[tri_str].append(ranks[rel_i])

            result_dict[mode + '_rank'].extend(ranks)
            result_dict[mode + '_counts'].extend(counts) # save count to normalize later


        return pred_to_gt, pred_5ples, rel_scores



###########################
def evaluate_recall(gt_rels, gt_boxes, gt_classes,
                    pred_rels, pred_boxes, pred_classes, rel_scores=None, cls_scores=None,
                    iou_thresh=0.5, phrdet=False):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    if pred_rels.size == 0:
        return [[]], np.zeros((0,5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)
    num_boxes = pred_boxes.shape[0]
    assert pred_rels[:,:2].max() < pred_classes.shape[0]

    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,1])
    assert np.all(pred_rels[:,2] > 0)

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(pred_rels[:,2], pred_rels[:,:2], pred_classes, pred_boxes,
                 rel_scores, cls_scores)

    # scores_overall = relation_scores.prod(1)
    # if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
    #     print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall[:10]))
        # Seems that it can be due to very similar score values
        # raise ValueError("Somehow the relations werent sorted properly")

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )

    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack((
        pred_rels[:,:2],
        pred_triplets[:, [0, 2, 1]],
    ))

    return pred_to_gt, pred_5ples, relation_scores


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-1) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-1), 2) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-1)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets: 
    :param pred_triplets: 
    :param gt_boxes: 
    :param pred_boxes: 
    :param iou_thresh: 
    :return: 
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]
            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def calculate_mR_from_evaluator_list(evaluator_list, mode, multiple_preds=False, save_file=None):
    """
    Copied from https://github.com/yuweihao/KERN/blob/master/lib/evaluation/sg_eval.py
    :param evaluator_list:
    :param mode:
    :param multiple_preds:
    :param save_file:
    :return:
    """
    all_rel_results = {}
    for (pred_id, pred_name, evaluator_rel) in evaluator_list:
        print('\n')
        print('relationship: ', pred_name)
        rel_results = evaluator_rel[mode].print_stats()
        all_rel_results[pred_name] = rel_results

    mean_recall = {}
    mR20 = 0.0
    mR50 = 0.0
    mR100 = 0.0
    mR200 = 0.0
    mR300 = 0.0
    for key, value in all_rel_results.items():
        if math.isnan(value['R@100']):# or math.isnan(value['R@300']):
            continue
        mR20 += value['R@20']
        mR50 += value['R@50']
        mR100 += value['R@100']
        mR200 += value['R@200']
        mR300 += value['R@300']
    rel_num = len(evaluator_list)
    mR20 /= rel_num
    mR50 /= rel_num
    mR100 /= rel_num
    mR200 /= rel_num
    mR300 /= rel_num
    mean_recall['R@20'] = mR20
    mean_recall['R@50'] = mR50
    mean_recall['R@100'] = mR100
    mean_recall['R@200'] = mR200
    mean_recall['R@300'] = mR300
    all_rel_results['mean_recall'] = mean_recall


    if multiple_preds:
        recall_mode = 'mean recall without constraint'
    else:
        recall_mode = 'mean recall with constraint'
    print('\n')
    print('======================' + mode + '  ' + recall_mode + '============================')
    print('mR@20: ', mR20)
    print('mR@50: ', mR50)
    print('mR@100: ', mR100)
    print('mR@200: ', mR200)

    if save_file is not None:
        if multiple_preds:
            save_file = save_file.replace('.pkl', '_multiple_preds.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(all_rel_results, f)

    return mean_recall


def eval_entry(mode, gt_entry, pred_entry, evaluator_list,
               evaluator_multiple_preds_list):

    for (pred_id, _, evaluator_rel), (_, _, evaluator_rel_mp) in zip(evaluator_list, evaluator_multiple_preds_list):
        gt_entry_rel = gt_entry.copy()
        mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
        gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
        if gt_entry_rel['gt_relations'].shape[0] == 0:
            continue

        evaluator_rel[mode].evaluate_scene_graph_entry(
            gt_entry_rel,
            pred_entry,
        )
        evaluator_rel_mp[mode].evaluate_scene_graph_entry(
            gt_entry_rel,
            pred_entry,
        )