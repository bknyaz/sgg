import h5py
import numpy as np
import os
import pickle


class VTESplit():
    '''
    Visual Translation Embedding Network for Visual Relation Detection
    Hanwang Zhang, Zawlin Kyaw, Shih-Fu Chang, Tat-Seng Chua
    '''
    def __init__(self, graphs_file, mode='train'):

        self.graphs_h5 = h5py.File(graphs_file, 'r')
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], self.mode
        self.corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']

        self.mode_ = ('test' if self.mode == 'test' else 'train')
        hdf5_path = 'gt/%s/' % self.mode_
        img_list = list(self.graphs_h5[hdf5_path].keys())

        print('VTE Split: read %d %s graphs' % (len(img_list), self.mode))

        pkl_file = graphs_file.replace('.h5', '_%s.pkl' % self.mode_)
        if os.path.exists(pkl_file):
            print('Loading from %s' % pkl_file)
            with open(pkl_file, 'rb') as f:
                self.data, self.cls, self.pre = pickle.load(f)
            print('{} scene graphs loaded'.format(len(self.data)))
        else:
            self.data = {}
            for image_id, img_key in enumerate(img_list):

                self.data[img_key] = {}

                # Borrowed from https://github.com/yangxuntu/vrd/blob/6c2e3f36129ea506f263efa34f95abd3e88a819c/tf-faster-rcnn-master/tools/vg_process_dete.py
                d = self.graphs_h5[hdf5_path + img_key]
                sub_box = d['sub_boxes'][:]
                obj_box = d['obj_boxes'][:]
                rlp_labels = d['rlp_labels'][:]
                sb = rlp_labels[:, 0]
                ob = rlp_labels[:, 2]
                if len(sub_box) == 0 or len(rlp_labels) == 0:
                    print('no bounding boxes', image_id, img_key)

                self.data[img_key]['boxes'], unique_inds, boxes_inds = np.unique(np.concatenate((sub_box, obj_box), axis=0),
                                                             axis=0, return_index=True, return_inverse=True)

                self.data[img_key]['gt_classes'] = np.concatenate((sb, ob), axis=0)[unique_inds]

                n = len(boxes_inds) // 2
                self.data[img_key]['rels'] = np.column_stack(
                    (boxes_inds[:n], boxes_inds[n:], rlp_labels[:, 1] + 1))  # +1 because the background will be added


            pre = list(self.graphs_h5['meta/pre/name2idx'].keys())
            pre.insert(0, '__background__')

            cls = list(self.graphs_h5['meta/cls/name2idx'].keys())

            cls[0], cls[1] = cls[1], cls[0]
            assert cls[0] == '__background__', cls

            self.cls = cls
            self.pre = pre

            self.graphs_h5.close()

            with open(pkl_file, 'wb') as f:
                pickle.dump((self.data, self.cls, self.pre), f)

        self.img_list = sorted(list(self.data.keys()))

        return

    def close(self):
        # TODO: implement with __exit__
        self.graphs_h5.close()

    def load_graphs(self, num_im=-1, num_val_im=0, filter_empty_rels=True, min_graph_size=-1, max_graph_size=-1,
                    training_triplets=None, random_subset=False, filter_zeroshots=True):

        img_list = self.img_list

        image_index = np.arange(len(img_list))
        if num_im > -1:
            image_index = image_index[:num_im]
        if num_val_im > 0:
            if self.mode in ['val']:  # , 'test' for faster preliminary evaluation on the test set
                image_index = image_index[:num_val_im]
            elif self.mode == 'train':
                image_index = image_index[num_val_im:]

        split_mask = np.zeros(len(img_list)).astype(bool)
        split_mask[image_index] = True

        boxes = []
        gt_classes = []
        relationships = []

        print('VTE Split: read %d %s graphs' % (len(image_index), self.mode))

        # Borrowed from https://github.com/yangxuntu/vrd/blob/6c2e3f36129ea506f263efa34f95abd3e88a819c/tf-faster-rcnn-master/tools/vg_process_dete.py
        # TODO: save preprocessed data for faster loading
        for image_id in image_index:

            basename = '{}.jpg'.format(img_list[image_id])
            if basename in self.corrupted_ims:
                split_mask[image_id] = 0
                continue

            boxes_i = self.data[img_list[image_id]]['boxes']
            gt_classes_i = self.data[img_list[image_id]]['gt_classes']
            rels = self.data[img_list[image_id]]['rels']

            if min_graph_size > -1 and len(gt_classes_i) <= min_graph_size:  # 0-10 will be excluded
                split_mask[image_id] = 0
                continue

            if max_graph_size > -1 and len(gt_classes_i) > max_graph_size:  # 11-Inf will be excluded
                split_mask[image_id] = 0
                continue

            assert len(boxes_i) == len(gt_classes_i), (len(boxes_i), len(gt_classes_i))
            assert filter_empty_rels, 'should filter empty rels'


            if len(gt_classes_i) < 2:  # since one object cannot have rels other than self-loops
                split_mask[image_id] = 0
                continue

            ind_zs = []
            if training_triplets:
                n = len(rels)
                if n > 0:
                    if random_subset:
                        ind_zs = np.random.permutation(n)[:int(np.round(n / 20.))]
                    else:
                        for rel_ind, tri in enumerate(rels):
                            o1, o2, R = tri
                            tri_str = '{}_{}_{}'.format(gt_classes_i[o1], R, gt_classes_i[o2])
                            if tri_str not in training_triplets:
                                ind_zs.append(rel_ind)

                    if filter_zeroshots:
                        if len(ind_zs) > 0:
                            ind_zs = np.array(ind_zs)
                            rels = rels[ind_zs]
                        else:
                            rels = np.zeros((0, 3), dtype=np.int32)


                if training_triplets and filter_zeroshots:
                    assert len(rels) == len(ind_zs), (len(rels), len(ind_zs))

                if filter_empty_rels and len(ind_zs) == 0:
                    split_mask[image_id] = 0
                    continue

            if filter_empty_rels and len(rels) == 0:
                split_mask[image_id] = 0
                continue

            boxes.append(boxes_i)
            gt_classes.append(gt_classes_i)
            relationships.append(rels)

        return split_mask, boxes, gt_classes, relationships


    def load_image_filenames(self, image_dir):
        """
        Loads the image filenames from visual genome
        :return: List of filenames corresponding to the good images
        """
        fns = []
        for i, img in enumerate(list(self.data.keys())):

            basename = '{}.jpg'.format(img)

            filename = os.path.join(image_dir, basename)
            if os.path.exists(filename):
                fns.append(basename)
        assert len(fns) == 25858 if self.mode == 'test' else 73794, len(fns)
        return fns

    def load_info(self):
        """
        Loads the file containing the visual genome label meanings
        :param info_file: JSON
        :return: ind_to_classes: sorted list of classes
                 ind_to_predicates: sorted list of predicates
        """

        return self.cls, self.pre