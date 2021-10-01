import numpy as np
import torch
from lib.pytorch_misc import enumerate_by_image
from lib.word_vectors import obj_edge_vectors


class SceneGraphPerturb():

    def __init__(self, method, embed_objs, subj_pred_obj_pairs,
                 L=0.2, topk=5, alpha=2, uniform=False, degree_smoothing=1,
                 data_dir=None, obj_classes=None, triplet2str=None):

        self.method = method
        if embed_objs is None:
            embed_objs = obj_edge_vectors(obj_classes,
                                          wv_dir=data_dir,
                                          wv_dim=200,
                                          avg_words=True)[0]
            embed_objs = embed_objs / torch.norm(embed_objs, 2, dim=1, keepdim=True)

        self.obj_pairwise = pairwise_similarity(embed_objs)
        self.subj_pred_obj_pairs = subj_pred_obj_pairs
        self.L = L
        self.topk = topk
        self.alpha = alpha
        self.uniform = uniform
        self.degree_smoothing = degree_smoothing
        self.n_obj_classes = self.obj_pairwise.shape[0]
        self.obj_classes = obj_classes
        self.triplet2str = triplet2str
        if self.method == 'neigh':
            assert self.topk > 0, self.topk


    def perturb(self, gt_obj, gt_rels, verbose=False):

        gt_obj_lst = [gt_obj[s:e] for _, s, e in enumerate_by_image(gt_obj[:, 0])]
        gt_rels_lst = [gt_rels[s:e] for _, s, e in enumerate_by_image(gt_rels[:, 0])]

        nodes = self.sample_nodes_(gt_obj_lst, gt_rels_lst)

        gt_obj_new = []
        for im, objs in enumerate(gt_obj_lst):  # for each image
            for obj_ind, obj_rels in zip(*nodes[im]):  # for each sampled node that will be perturbed

                if verbose:
                    before = objs[obj_ind, 1]
                    print('\nbefore: %s' % self.obj_classes[before])
                    for (_, o1, o2, R) in obj_rels:
                        print(self.triplet2str('{}_{}_{}'.format(objs[o1, 1], R, objs[o2, 1])))

                objs[obj_ind, 1] = self.perturb_object_(objs, obj_rels, obj_ind, verbose=verbose)

                if verbose:
                    print('\nafter: %s' % self.obj_classes[objs[obj_ind, 1]])
                    for (_, o1, o2, R) in obj_rels:
                        print(self.triplet2str('{}_{}_{}'.format(objs[o1, 1], R, objs[o2, 1])))

            gt_obj_new.append(objs)

        gt_obj_new = torch.cat(gt_obj_new)

        return gt_obj_new


    def perturb_object_(self, gt_objs, gt_rels, ind, verbose=False):

        cls = gt_objs[ind, 1].item()

        if self.method == 'rand':
            candidates = torch.cat((torch.arange(1, cls), torch.arange(cls + 1, self.n_obj_classes)))
            cls_new = random_choice(candidates)  # random class except for background and the same class (cls)

        elif self.method == 'neigh':

            candidates = torch.argsort(self.obj_pairwise[cls])[-self.topk:]
            cls_new = random_choice(candidates)

        elif self.method == 'graphn':
            all_candidates = {}
            for (_, o1, o2, R) in gt_rels:  # person_on_surfboard, wave_near_person
                assert ind in [o1, o2], (ind, o1, o2, R)
                if ind == o1:
                    pair = '{}_{}'.format(R, gt_objs[o2, 1])  # what is on surfboard?
                    pairs = self.subj_pred_obj_pairs[1]  # all pred_obj_pairs
                else:
                    pair = '{}_{}'.format(gt_objs[o1, 1], R)  # wave near what?
                    pairs = self.subj_pred_obj_pairs[0]  # all subj_pred_pairs

                if pair in pairs:
                    for obj, freq in pairs[pair].items():
                        if obj != cls:  # skip candidates of the same class as the original object label
                            if obj not in all_candidates:
                                all_candidates[obj] = []
                            all_candidates[obj].append(freq)

            candidates, probs = [], []
            for obj in all_candidates:
                # e.g. person_on_surfboard -> dog_on_surfboard
                freq = np.array(all_candidates[obj])  # e.g. for a dog: 10 times (wave_near_dog), 30 times (person_near_dog)
                if len(freq) >= max(1, min(len(gt_rels), 2)) and np.min(freq) >= self.alpha:
                    # using the obj class instead of cls must result in at least 2 possible triplets and all resulted triplets must be alpha-shots or more
                    candidates.append(obj)
                    probs.append(np.mean(freq))  # frequency in the dataset


            if len(candidates) == 0:
                if verbose:
                    print('\nzero candidates !!!')
                cls_new = cls
            else:
                probs = 1 / np.array(probs)     # inverse of the frequency in the dataset
                probs = probs / np.sum(probs)   # normalize to have probabilities
                cls_new = np.random.choice(candidates, p=probs)

                if verbose:
                    print('candidate: prob')
                    candidates = np.array(candidates)
                    for c, p in zip(candidates[:5], probs[:5]):
                        print('{:15s}: {:.5f}'.format(self.obj_classes[c], p))
                    best = np.argmax(probs)
                    print('max:', self.obj_classes[candidates[best]], '%.5f' % probs[best])
                    print('chosen:', self.obj_classes[cls_new], '%.5f' % probs[np.where(candidates == cls_new)[0]])

                assert cls_new not in [0, cls], (cls_new, cls)

            if self.topk > 0:
                # Choose from top-k semantic neighbors of cls_new
                obj_pairwise_new = self.obj_pairwise[cls_new].clone()
                obj_pairwise_new[cls_new] = np.Inf  # including cls_new
                obj_pairwise_new[cls] = -np.Inf     # excluding cls
                # Return topk+1 candidates: topk semantic neighbors and cls_new itself
                candidates = torch.argsort(obj_pairwise_new)[-self.topk - 1:]
                if verbose:
                    print('semantic neighbors:', ', '.join([self.obj_classes[c] for c in candidates]))

                cls_new = random_choice(candidates)

        else:
            raise NotImplementedError(self.method)

        if not (self.method == 'graphn' and self.topk == 0):
            assert cls_new not in [0, cls], (cls_new, cls)

        return cls_new


    def sample_nodes_(self, gt_obj_lst, gt_rels_lst):
        nodes = {}
        for im, (objs, rels) in enumerate(zip(gt_obj_lst, gt_rels_lst)): # per image
            ind_fg = rels[:, -1] > 0
            node2rels, degrees = [], []
            n_nodes = len(objs)
            for obj in range(n_nodes):
                ind_rels = torch.nonzero( ind_fg & ((rels[:, 1] == obj) |
                                                    (rels[:, 2] == obj))).view(-1)
                node2rels.append(rels[ind_rels])
                degrees.append(len(node2rels[-1]))

            if self.L <= 0:
                nodes[im] = (np.empty(0), [])
            else:
                if self.uniform:
                    probs = np.ones(n_nodes, dtype=np.float32)
                else:
                    probs = np.array(degrees, dtype=np.float32) ** self.degree_smoothing  # flatten the distribution a bit
                    probs = probs.clip(1e-2, None)  # replacing 0 with a small value to allow sampling isolate nodes sometimes

                probs = probs / np.sum(probs)  # actual probs

                nodes_max = max(1, int(np.round(self.L * n_nodes)))  # at least one node per image
                node_inds = np.random.choice(np.arange(n_nodes), size=nodes_max, replace=False, p=probs)
                nodes[im] = (node_inds, [node2rels[i] for i in node_inds])

                if self.L >= 1:
                    assert len(node_inds) == n_nodes == nodes_max, (len(node_inds), n_nodes, nodes_max, probs)

        return nodes



def pairwise_similarity(embed_objs):
    obj_pairwise = torch.mm(embed_objs, embed_objs.t())
    obj_pairwise[0, :] = -np.Inf  # to avoid choosing the background
    obj_pairwise[:, 0] = -np.Inf  # to avoid choosing the background
    obj_pairwise[np.diag_indices_from(obj_pairwise)] = -np.Inf  # to avoid choosing the same object
    return obj_pairwise


def random_choice(tensor):
    return tensor[torch.randperm(len(tensor))[0]].item()
