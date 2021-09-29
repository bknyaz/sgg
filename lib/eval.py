import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import IM_SCALE, BOX_SCALE, NO_GRAD
from lib.sgg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, eval_entry
from lib.pytorch_misc import set_mode, argsort_desc
from lib.visualize import show_nx, draw_boxes
from dataloaders.visual_genome import VG
from lib.get_dataset_counts import get_counts


all_shot_splits = ['val_alls', 'test_alls']


def val_epoch(mode, sgg_model, loader, name, triplet_counts, triplet2str, n_batches=-1, is_test=False, save_scores=False,
              predicate_weight=0, train=None, wandb_log=None, **kwargs):
    print('\nEvaluate %s %s triplets' % (name.upper(), 'test' if is_test else 'val'))
    sgg_model.eval()
    evaluator, all_pred_entries, all_metrics = {}, {}, []

    EVAL_MODES = ['sgdet'] if mode == 'sgdet' else ['predcls', 'sgcls']
    assert mode in EVAL_MODES, (mode, 'other modes not supported')

    predicate_weights = None
    if predicate_weight != 0:
        fg_matrix, bg_matrix = get_counts(train, must_overlap=True)
        fg_matrix[:, :, 0] = bg_matrix + 1
        fg_matrix = fg_matrix + 1
        predicate_weights = fg_matrix.mean(axis=(0, 1)) ** predicate_weight


    with NO_GRAD():
        for eval_m in EVAL_MODES:
            if eval_m == 'sgdet' and name.find('val_') >= 0:
                continue  # skip for validation, because it takes a lot of time

            print('\nEvaluating %s...' % eval_m.upper())

            evaluator[eval_m] = BasicSceneGraphEvaluator(eval_m)  # graph constrained evaluator
            evaluator[eval_m + '_nogc'] = BasicSceneGraphEvaluator(eval_m, multiple_preds=True,    # graph unconstrained evaluator
                                                                   per_triplet=name in all_shot_splits,
                                                                   triplet_counts=triplet_counts,
                                                                   triplet2str=triplet2str)

            # for calculating recall of each relationship except no relationship
            evaluator_list, evaluator_multiple_preds_list = [], []
            if name not in ['val_zs', 'test_zs'] and name.find('val_') < 0:
                for index, name_s in enumerate(loader.dataset.ind_to_predicates):
                    if index == 0:
                        continue
                    evaluator_list.append((index, name_s, BasicSceneGraphEvaluator.all_modes()))
                    evaluator_multiple_preds_list.append(
                        (index, name_s, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))


            set_mode(sgg_model, mode=eval_m, is_train=False, verbose=True)


            # For all val/test batches
            all_pred_entries[eval_m] = []
            for val_b, batch in enumerate(tqdm(loader)):
                pred_entry = val_batch(sgg_model, val_b, batch,
                                       evaluator, eval_m, loader.dataset, evaluator_list, evaluator_multiple_preds_list,
                                       train=train, predicate_weights=predicate_weights, **kwargs)
                if save_scores:
                    all_pred_entries[eval_m].extend(pred_entry)

                if n_batches > -1 and val_b + 1 >= n_batches:
                    break


            evaluator[eval_m].print_stats()
            evaluator[eval_m + '_nogc'].print_stats()

            mean_recall = mean_recall_mp = None
            if len(evaluator_list) > 0:
                # Compute Mean Recall Results
                mean_recall = calculate_mR_from_evaluator_list(evaluator_list, eval_m, save_file=None)
                mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, eval_m,
                                                                  multiple_preds=True, save_file=None)

            if not wandb_log:
                continue

            # Log using WANDB
            eval_gc = evaluator[eval_m].result_dict
            eval_no_gc = evaluator[eval_m + '_nogc'].result_dict
            results_dict = {}
            for eval_, mean_eval, sfx in zip([eval_gc, eval_no_gc], [mean_recall, mean_recall_mp], ['GC', 'NOGC']):
                for k, v in eval_[eval_m + '_recall'].items():
                    all_metrics.append(np.mean(v))
                    results_dict['%s/%s_R@%i_%s' % (eval_m, name, k, sfx)] = np.mean(v)
                if mean_eval:
                    for k, v in mean_eval.items():
                        results_dict['%s/%s_m%s_%s' % (eval_m, name, k, sfx)] = np.mean(v)

            # Per triplet metrics
            try:
                if name in all_shot_splits:
                    for case in ['', '_norm']:
                        for k, v in eval_no_gc[eval_m + '_recall_triplet' + case].items():
                            results_dict['%s/%s_R@%i_triplet%s' % (eval_m, name, k, case)] = v
                        for metric in ['meanrank', 'medianrank'] + (['medianrankclass'] if case == '' else []):
                            results_dict['%s/%s_%s_triplet%s' % (eval_m, name, metric, case)] = \
                                eval_no_gc[eval_m + ('_%s_triplet' % metric) + case]
            except Exception as e:
                print('error in per triplet eval', e)

            wandb_log(results_dict, step=sgg_model.global_batch_iter,
                      is_summary=True, log_repeats=5 if is_test else 1)


    if wandb_log:
        wandb_log({'avg/%s_R' % (name): np.mean(all_metrics)}, step=sgg_model.global_batch_iter,
                  is_summary=True, log_repeats=5 if is_test else 1)

    return all_pred_entries


def val_batch(sgg_model, batch_num, b, evaluator, eval_m, val_dataset, evaluator_list, evaluator_multiple_preds_list,
              vis=False, max_obj=10, max_rels=20, train=None, test_zs=None, predicate_weights=None):

    if val_dataset.torch_detector:
        scale = 1.
        box_threshs = [0.2, 0.05, 0.01]
    else:
        scale = BOX_SCALE / IM_SCALE
        box_threshs = [None]

    pred_entries = []
    for box_score_thresh in box_threshs:
        sgg_model.set_box_score_thresh(box_score_thresh)
        try:
            det_res = sgg_model(b.scatter())  # keep as it was in the original code

            det_res = [det_res]

            for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):

                if vis and len(val_dataset.gt_classes[batch_num + i]) > max_obj:
                    print('skipping a scene graph with too many objects')
                    continue

                if VG.split == 'stanford':
                    w, h = b[i][1][0, :2]
                    scale_gt = 1. / (BOX_SCALE / max(w, h))
                else:
                    scale_gt = 1.

                gt_entry = {
                    'gt_classes': val_dataset.gt_classes[batch_num + i].copy(),
                    'gt_relations': val_dataset.relationships[batch_num + i].copy(),
                    'gt_boxes': val_dataset.gt_boxes[batch_num + i].copy() * scale_gt,
                }

                pred_entry = {
                    'pred_boxes': boxes_i * scale,
                    'pred_classes': objs_i,
                    'pred_rel_inds': rels_i,
                    'obj_scores': obj_scores_i,
                    'rel_scores': pred_scores_i,  # hack for now.
                }

                if predicate_weights is not None:
                    p = 1. / predicate_weights[1:]
                    pred_entry['rel_scores'][:, 1:] = pred_entry['rel_scores'][:, 1:] * p
                    pred_entry['rel_scores'] = pred_entry['rel_scores'] / np.sum(pred_entry['rel_scores'], axis=1, keepdims=True)
                    assert (abs(pred_entry['rel_scores'].sum(1) - 1) < 1e-5).all(), pred_entry['rel_scores'].sum(1)

                pred_entries.append(pred_entry)

                for sfx in ['', '_nogc']:
                    evaluator[eval_m + sfx].evaluate_scene_graph_entry(
                        gt_entry,
                        pred_entry
                    )

                if evaluator_list is not None and len(evaluator_list) > 0:
                    eval_entry(eval_m, gt_entry, pred_entry,
                               evaluator_list, evaluator_multiple_preds_list)

                if vis:
                    print(val_dataset.filenames[batch_num + i], 'showing ground truth')
                    im_gt = draw_boxes(b[0][0][0].permute(1, 2, 0).data.cpu().numpy().copy(),
                                       [train.ind_to_classes[c] for c in gt_entry['gt_classes']],
                                       gt_entry['gt_boxes'],
                                       torch_detector=val_dataset.torch_detector)
                    plt.figure(figsize=(10, 10))
                    plt.imshow(im_gt)
                    plt.axis(False)
                    plt.show()

                    show_nx(gt_entry['gt_classes'],
                                 gt_entry['gt_boxes'],
                                 gt_entry['gt_relations'],
                                 train_set=train,
                                 test_set=test_zs,
                            torch_detector=val_dataset.torch_detector)

                    print(val_dataset.filenames[batch_num + i], 'showing top %d relationships' % max_rels)
                    im_pred = draw_boxes(b[0][0][0].permute(1, 2, 0).data.cpu().numpy().copy(),
                                         [train.ind_to_classes[c] for c in pred_entry['pred_classes']],
                                         pred_entry['pred_boxes'],
                                         torch_detector=val_dataset.torch_detector)
                    plt.figure(figsize=(10, 10))
                    plt.imshow(im_pred)
                    plt.axis(False)
                    plt.show()

                    obj_scores = pred_entry['obj_scores']
                    pred_rel_inds = rels_i
                    obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
                    overall_scores = obj_scores_per_rel[:, None] * pred_entry['rel_scores'][:, 1:]
                    score_inds = argsort_desc(overall_scores)[:max_rels]
                    pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1] + 1))

                    show_nx(pred_entry['pred_classes'],
                            pred_entry['pred_boxes'],
                            pred_rels,
                            train_set=train,
                            test_set=test_zs,
                            torch_detector=val_dataset.torch_detector)


            return pred_entries

        except (ValueError, IndexError) as e:
            print('no objects or relations found'.upper(), e, b[0][-1], 'trying a smaller threshold')
