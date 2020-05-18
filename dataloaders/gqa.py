"""
Data loading functions for the GQA dataset: https://cs.stanford.edu/people/dorarad/gqa/about.html
"""

import os
import numpy as np


def load_image_filenames(image_ids, mode, image_dir):
    """
    Loads the image filenames from GQA from the JSON file that contains them.
    :param image_file: JSON file. Elements contain the param "image_id".
    :param image_dir: directory where the GQA images are located
    :return: List of filenames corresponding to the good images
    """
    fns = []
    for im_id in image_ids:
        basename = '{}.jpg'.format(im_id)
        filename = os.path.join(image_dir, basename)
        if os.path.exists(filename):  # comment for faster loading
            fns.append(basename)

    assert len(fns) == len(image_ids), (len(fns), len(image_ids))
    assert len(fns) == (72140 if mode in ['train', 'val'] else 10234), (len(fns), mode)
    return fns


def load_graphs(all_sgs_json, image_ids, classes_to_ind, predicates_to_ind, num_val_im=-1,
                min_graph_size=-1, max_graph_size=-1, mode='train',
                training_triplets=None, random_subset=False,
                filter_empty_rels=True, filter_zeroshots=True,
                exclude_left_right=False):
    """
    Load GT boxes, relations and dataset split
    :param graphs_file_template: template SG filename (replace * with mode)
    :param split_modes_file: JSON containing mapping of image id to its split
    :param mode: (train, val, or test)
    :param training_triplets: a list containing triplets in the training set
    :param random_subset: whether to take a random subset of relations as 0-shot
    :param filter_empty_rels: (will be filtered otherwise.)
    :return: image_index: a np array containing the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    if exclude_left_right:
        print('\n excluding some relationships from GQA!\n')
        filter_rels = []
        for rel in ['to the left of', 'to the right of']:
            filter_rels.append(predicates_to_ind[rel])
        filter_rels = set(filter_rels)


    # Load the image filenames split (i.e. image in train/val/test):
    # train - 0, val - 1, test - 2
    image_index = np.arange(len(image_ids))  # all training/test images
    if num_val_im > 0:
        if mode in ['val']:
            image_index = image_index[:num_val_im]
        elif mode == 'train':
            image_index = image_index[num_val_im:]

    split_mask = np.zeros(len(image_ids)).astype(np.bool)
    split_mask[image_index] = True

    print(mode, np.sum(split_mask))

    image_idxs = {}
    for i, imid in enumerate(image_ids):
        image_idxs[imid] = i

    # Get everything by SG
    boxes = []
    gt_classes = []
    relationships = []
    for imid in image_ids:

        if not split_mask[image_idxs[imid]]:
            continue

        sg_objects = all_sgs_json[imid]['objects']
        # Sort the keys to ensure object order is always the same
        sorted_oids = sorted(list(sg_objects.keys()))

        assert filter_empty_rels, 'should filter images with empty rels'

        # Filter out images without objects/bounding boxes
        if len(sorted_oids) == 0:
            split_mask[image_idxs[imid]] = False
            continue

        boxes_i = []
        gt_classes_i = []
        raw_rels = []
        oid_to_idx = {}
        no_objs_with_rels = True
        for oid in sorted_oids:

            obj = sg_objects[oid]

            # Compute object GT bbox
            b = np.array([obj['x'], obj['y'], obj['w'], obj['h']])
            try:
                assert np.all(b[:2] >= 0), (b, obj)  # sanity check
                assert np.all(b[2:] > 0), (b, obj)  # no empty box
            except:
                continue  # skip objects with empty bboxes or negative values


            oid_to_idx[oid] = len(gt_classes_i)
            if len(obj['relations']) > 0:
                no_objs_with_rels = False

            # Compute object GT class
            gt_class = classes_to_ind[obj['name']]
            gt_classes_i.append(gt_class)

            # convert to x1, y1, x2, y2
            box = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])

            # box = np.concatenate((b[:2] - b[2:] / 2, b[:2] + b[2:] / 2))
            boxes_i.append(box)

            # Compute relations from this object to others in the current SG
            for rel in obj['relations']:
                raw_rels.append([oid, rel['object'], rel['name']])  # s, o, r

        # Filter out images without relations - TBD
        if no_objs_with_rels:
            split_mask[image_idxs[imid]] = False
            continue

        if min_graph_size > -1 and len(gt_classes_i) <= min_graph_size:  # 0-10 will be excluded
            split_mask[image_idxs[imid]] = False
            continue

        if max_graph_size > -1 and len(gt_classes_i) > max_graph_size:  # 11-Inf will be excluded
            split_mask[image_idxs[imid]] = False
            continue

        # Update relations to include SG object ids
        rels = []
        for rel in raw_rels:
            if rel[0] not in oid_to_idx or rel[1] not in oid_to_idx:
                continue   # skip rels for objects with empty bboxes

            R = predicates_to_ind[rel[2]]

            if exclude_left_right:
                if R in filter_rels:
                    continue

            rels.append([oid_to_idx[rel[0]],
                         oid_to_idx[rel[1]],
                         R])

        rels = np.array(rels)
        n = len(rels)
        if n == 0:
            split_mask[image_idxs[imid]] = False
            continue

        elif training_triplets:
            if random_subset:
                ind_zs = np.random.permutation(n)[:int(np.round(n/15.))]
            else:
                ind_zs = []
                for rel_ind, tri in enumerate(rels):
                    o1, o2, R = tri
                    tri_str = '{}_{}_{}'.format(gt_classes_i[o1],
                                                R,
                                                gt_classes_i[o2])
                    if tri_str not in training_triplets:
                        ind_zs.append(rel_ind)
                        # print('%s not in the training set' % tri_str, tri)
                ind_zs = np.array(ind_zs)

            if filter_zeroshots:
                if len(ind_zs) > 0:
                    try:
                        rels = rels[ind_zs]
                    except:
                        print(len(rels), ind_zs)
                        raise
                else:
                    rels = np.zeros((0, 3), dtype=np.int32)

            if filter_empty_rels and len(ind_zs) == 0:
                split_mask[image_idxs[imid]] = False
                continue

        # Add current SG information to the dataset
        boxes_i = np.array(boxes_i)
        gt_classes_i = np.array(gt_classes_i)

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, relationships


def load_info(train_sgs, val_sgs):
    """
    Loads the file containing the GQA label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
             classes_to_ind: map from object classes to indices
             predicates_to_ind: map from predicate classes to indices
    """
    info = {'label_to_idx': {}, 'predicate_to_idx': {}}

    obj_classes = set()
    for sg in list(train_sgs.values()) + list(val_sgs.values()):
        for obj in sg['objects'].values():
            obj_classes.add(obj['name'])
    ind_to_classes = ['__background__'] + sorted(list(obj_classes))
    for obj_lbl, name in enumerate(ind_to_classes):
        info['label_to_idx'][name] = obj_lbl

    rel_classes = set()
    for sg in list(train_sgs.values()) + list(val_sgs.values()):
        for obj in sg['objects'].values():
            for rel in obj['relations']:
                rel_classes.add(rel['name'])
    ind_to_predicates = ['__background__'] + sorted(list(rel_classes))
    for rel_lbl, name in enumerate(ind_to_predicates):
        info['predicate_to_idx'][name] = rel_lbl

    assert info['label_to_idx']['__background__'] == 0
    assert info['predicate_to_idx']['__background__'] == 0

    return (ind_to_classes, ind_to_predicates,
            info['label_to_idx'], info['predicate_to_idx'])
