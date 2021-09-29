import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from torch.nn.functional import binary_cross_entropy_with_logits as BCE

from lib.pytorch_misc import enumerate_by_image
from lib.word_vectors import obj_edge_vectors
from .graphconv import GraphTripleConvNet
from .layout import boxes_to_layout
from .crn import RefinementNetwork


class GAN(nn.Module):
    """
    GAN model that can be added to the models that generate (predict) scene graphs from images,
    such as RelModel or RelModelStanford.

    B. Knyazev, H. de Vries, C. Cangea, G.W. Taylor, A. Courville, E. Belilovsky.
    Generative Compositional Augmentations for Scene Graph Prediction. ICCV 2021.
    https://arxiv.org/abs/2007.05756
    """
    def __init__(self,
                 obj_classes,
                 rel_classes,
                 embed_dim=200,
                 hidden_dim=64,
                 n_ch=512,
                 pool_sz=7,
                 fmap_sz=38,
                 losses=('D', 'G', 'rec'),
                 SN=True,
                 BN=True,
                 n_layers_G=5,
                 vis_cond=None,
                 init_embed=False,
                 largeD=False,
                 data_dir='',  # to load word embeddings
                 device='cuda'):

        """
        :param embed_dim: Dimension for all embeddings
        :param obj_dim:
        """
        super(GAN, self).__init__()
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.embed_dim = embed_dim
        self.n_ch = n_ch
        self.obj_dim = pool_sz ** 2 * n_ch
        self.pool_sz = pool_sz
        self.fmap_sz = fmap_sz
        self.losses = losses
        self.SN = SN
        self.BN = BN
        self.vis_cond = vis_cond
        self.largeD = largeD
        self.h5_data = None
        self.device = device
        if vis_cond is not None:
            self.h5_data = h5py.File(vis_cond, mode='r')

        self.G_obj_embed = nn.Embedding(len(self.obj_classes), self.embed_dim)
        self.G_rel_embed = nn.Embedding(len(self.rel_classes), self.embed_dim)

        if SN:
            conv = lambda n_in, n_out, ks, pad: spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=ks, padding=pad))
        else:
            conv = lambda n_in, n_out, ks, pad: nn.Conv2d(n_in, n_out, kernel_size=ks, padding=pad)

        def cond_discriminator(n_classes):                                  # input is 512x7x7
            return nn.Sequential(conv(n_ch + n_classes, n_ch // 2, 3, 0),   # ->256x5x5
                                 nn.ReLU(),
                                 conv(n_ch // 2, n_ch // 4, 3, 0),          # ->128x3x3
                                 nn.ReLU(),
                                 conv(n_ch // 4, n_ch // 8, 1, 0),          # ->64x3x3
                                 nn.ReLU(),
                                 conv(n_ch // 8, 1, 3, 0),                  # ->1x1x1
                                 nn.Flatten())

        # Discriminators (must start with D_) ----------------------------------
        self.D_nodes = cond_discriminator(len(self.obj_classes))
        self.D_edges = cond_discriminator(len(self.rel_classes))
        self.D_global = nn.Sequential(conv(n_ch, n_ch // 2, 3, 0),                                          # 512x38x38->256x36x36
                                      nn.LeakyReLU(0.2),
                                      conv(n_ch // 2, n_ch // 2, 1, 0) if largeD else nn.Identity(),        # ->256x36x36
                                      nn.LeakyReLU(0.2) if largeD else nn.Identity(),
                                      nn.AvgPool2d(2, ceil_mode=True) if fmap_sz > 24 else nn.Identity(),   # ->256x18x18
                                      conv(n_ch // 2, n_ch // 2, 3, 0),                                     # ->256x16x16
                                      nn.LeakyReLU(0.2),
                                      conv(n_ch // 2, n_ch // 2, 1, 0) if largeD else nn.Identity(),        # ->256x16x16
                                      nn.LeakyReLU(0.2) if largeD else nn.Identity(),
                                      nn.AvgPool2d(2),                                                      # ->256x8x8
                                      conv(n_ch // 2, n_ch // 4, 3, 0),                                     # ->128x6x6
                                      nn.LeakyReLU(0.2),
                                      conv(n_ch // 4, n_ch // 4, 1, 0) if largeD else nn.Identity(),        # ->128x6x6
                                      nn.LeakyReLU(0.2) if largeD else nn.Identity(),
                                      nn.AvgPool2d(2),                                                      # ->128x3x3
                                      conv(n_ch // 4, 1, 3, 0),                                             # ->128x1x1
                                      nn.Flatten())
        print('Global Discriminator:', self.D_global)

        # Generators (must start with G_) --------------------------------------

        # Graph Convolutional Network (returns 32x7x7 features)
        self.G_gcn = GraphTripleConvNet(input_dim=self.embed_dim + 4,
                                      input_edge_dim=self.embed_dim,
                                      output_dim=hidden_dim // 2 * pool_sz * pool_sz,
                                      num_layers=n_layers_G,
                                      hidden_dim=hidden_dim,
                                      pooling='avg',
                                      mlp_normalization='batch' if BN else 'none')

        # Post process GCN features with conv layers to make them more "spatial"
        self.G_node = nn.Sequential(nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                                    nn.ReLU())

        # To transform hidden features concatenated with visual features
        self.G_proj = nn.Conv2d(hidden_dim + int(vis_cond is not None) * n_ch, hidden_dim, kernel_size=1)

        # To generate large global features
        self.G_refine = RefinementNetwork(dims=(hidden_dim, n_ch // 4, n_ch // 2, n_ch), 
                                          normalization='batch', 
                                          activation='leakyrelu-0.2')

        # Predefine 0,1 labels to prevent resource consuming creation of large vectors for every batch during training
        n_max = 50000  # some random big number
        self.y_real_, self.y_fake_ = torch.ones(n_max, 1).to(device), torch.zeros(n_max, 1).to(device)
        self.y_real = lambda n: Variable(self.y_real_[:n])
        self.y_fake = lambda n: Variable(self.y_fake_[:n])

        # Load the Glove-based language model to use for SG perturbations and initializing the GAN input embeddings
        embed_objs, word_vectors = obj_edge_vectors(self.obj_classes,
                                                    wv_dir=data_dir,
                                                    wv_dim=embed_dim,
                                                    word_vectors=None,
                                                    avg_words=True)

        self.embed_objs = (embed_objs / torch.norm(embed_objs, 2, dim=1, keepdim=True)).to(device)

        if init_embed:
            # Initialize learnable GAN embeddings with the Glove ones
            # Using this led to worse results in our experiments, so we don't use it
            assert self.G_obj_embed.weight.shape == self.embed_objs.shape, (self.G_obj_embed.weight.shape, self.embed_objs.shape)
            self.G_obj_embed.weight.data = self.embed_objs.clone()

            embed_rels = obj_edge_vectors(self.rel_classes,
                                          wv_dim=embed_dim,
                                          word_vectors=word_vectors,
                                          avg_words=True)[0]

            self.embed_rels = (embed_rels / torch.norm(embed_rels, 2, dim=1, keepdim=True)).to(device)
            assert self.G_rel_embed.weight.shape == self.embed_rels.shape, (self.G_rel_embed.weight.shape, self.embed_rels.shape)
            self.G_rel_embed.weight.data = self.embed_rels.clone()


    def loss_fn(self, predictions, is_fake=True, updateD=False):
        if updateD:
            if is_fake:
                loss = BCE(predictions, self.y_fake(len(predictions)))
            else:
                loss = BCE(predictions, self.y_real(len(predictions)))
        else:
            assert is_fake
            loss = BCE(predictions, self.y_real(len(predictions)))
        return loss


    def forward(self, gt_objects, boxes_scaled, gt_rels):

        gt_objects, boxes_scaled, gt_rels = dummy_nodes(gt_objects, boxes_scaled, gt_rels)

        obj_vecs = self.G_obj_embed(gt_objects[:, -1])
        pred_vecs = self.G_rel_embed(gt_rels[:, -1])

        # GCN forward pass
        obj_fg = torch.nonzero(gt_objects[:, -1]).view(-1)
        nodes_fake = self.G_gcn(torch.cat((obj_vecs, boxes_scaled), dim=1), pred_vecs, gt_rels[:, 1:3])[0][obj_fg]

        n_obj = len(obj_fg)
        gt_objects, boxes = gt_objects[obj_fg], boxes_scaled[obj_fg]
        assert len(nodes_fake) == len(gt_objects) == len(boxes), (nodes_fake.shape, gt_objects.shape, boxes.shape)

        # Make features more "spatial"
        nodes_fake = self.G_node(nodes_fake.view(n_obj, -1, self.pool_sz, self.pool_sz))

        if self.h5_data is not None:
            vis_features = []
            for i, cls in enumerate(gt_objects[:, -1].data.cpu().numpy()):
                assert cls > 0 , 'background objects are not expected here'
                dset = self.h5_data[self.obj_classes[cls]]
                ind = np.random.permutation(dset.shape[0])[0]
                vis_features.append(torch.from_numpy(dset[ind]).view(1, self.n_ch, self.pool_sz, self.pool_sz))
            nodes_fake = torch.cat((torch.cat(vis_features).to(nodes_fake), nodes_fake), dim=1)

        # Generate global feature maps from visual features
        fmap_fake = F.relu(self.G_refine(boxes_to_layout(self.G_proj(nodes_fake),
                                                         boxes,
                                                         gt_objects[:, 0],
                                                         self.fmap_sz,
                                                         self.fmap_sz,
                                                         pooling='sum')))
        return fmap_fake


    def loss(self, features_real=None, features_fake=None, is_nodes=False, updateD=False, labels_fake=None, labels_real=None, is_fmaps=False):
        '''
        Same for nodes and edges, just different neural net to use
        :param features_real:
        :param features_fake:
        :param indices: should be [interv_ind, noninterv_ind]
        :return:
        '''
        if updateD and 'D' not in self.losses:
            return {}
        elif not updateD and 'G' not in self.losses:
            return {}

        if not is_fmaps:
            # Label conditioning
            n_fake = len(features_fake)
            n_classes = len(self.obj_classes if is_nodes else self.rel_classes)
            y_vec_fake = torch.zeros(n_fake, n_classes, device=self.device).scatter_(1, labels_fake.view(-1, 1), 1)
            y_fill_fake = y_vec_fake.unsqueeze(2).unsqueeze(3).expand(n_fake, -1, self.pool_sz, self.pool_sz)
            features_fake = features_fake.view(n_fake, -1, self.pool_sz, self.pool_sz)
            features_fake = torch.cat([features_fake, y_fill_fake], 1)

            if updateD:
                n_real = len(features_real)
                if labels_real is None:
                    y_fill_real = y_fill_fake.clone()
                else:
                    y_vec_real = torch.zeros(n_real, n_classes, device=self.device).scatter_(1, labels_real.view(-1, 1), 1)
                    y_fill_real = y_vec_real.unsqueeze(2).unsqueeze(3).expand(n_real, -1, self.pool_sz, self.pool_sz)

                features_real = features_real.view(n_real, -1, self.pool_sz, self.pool_sz)
                features_real = torch.cat([features_real, y_fill_real], 1)

        fn = self.D_global if is_fmaps else (self.D_nodes if is_nodes else self.D_edges)

        if not updateD:
            assert labels_real is None and features_real is None, 'do not need real labels/features in case of G update'

        # detach from feature map extraction
        real_loss = self.loss_fn(fn(features_real.detach()), is_fake=False, updateD=True) if updateD else 0

        # detach from G (even if different optimizers are used, it should save compute)
        fake_loss = self.loss_fn(fn(features_fake.detach() if updateD else features_fake), is_fake=True, updateD=updateD)

        key = '_'.join(('D' if updateD else 'G',
                        'fmap' if is_fmaps else ('obj' if is_nodes else 'rel')))
        loss = { key: real_loss + fake_loss }

        return loss


def dummy_nodes(gt_objs, gt_boxes, gt_rels):
    # Add dummy nodes to scene graphs to improve message propagation
    gt_objs_new, gt_boxes_new, gt_rels_new = [], [], []

    gt_rels_lst = [gt_rels[s:e] for im, s, e in enumerate_by_image(gt_rels[:, 0])]
    dummy_box = torch.Tensor([0, 0, 1, 1]).view(1, 4).to(gt_boxes)
    offset = 0
    for im, s, e in enumerate_by_image(gt_objs[:, 0]):
        gt_objs_im = gt_objs[s:e]
        n_obj = len(gt_objs_im)

        rels = torch.zeros((n_obj * 2, 4)).to(gt_rels)  # adding two way edges from/to the dummy node
        rels[:, 0] = im
        for i in range(n_obj):
            # make edges two way as in the visual genome data loader
            for j, in_out in zip([i, i + n_obj], [(1, 2), (2, 1)]):
                rels[j, in_out[0]] = n_obj
                rels[j, in_out[1]] = i

        rels = torch.cat((gt_rels_lst[im].clone(), rels), 0)
        rels[:, 1:3] += offset
        gt_rels_new.append(rels)
        gt_objs_new.append(torch.cat((gt_objs_im, torch.Tensor([im, 0]).view(1, 2).to(gt_objs_im)), 0))
        gt_boxes_new.append(torch.cat((gt_boxes[s:e], dummy_box), 0))
        offset += (n_obj + 1)  # +1 because 1 dummy node is added
        # assert len(torch.cat(gt_objs_new)) == offset, (torch.cat(gt_objs_new).shape, offset)

    return torch.cat(gt_objs_new), torch.cat(gt_boxes_new), torch.cat(gt_rels_new)
