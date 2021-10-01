"""
Configuration file!
"""

import os
from argparse import ArgumentParser
import subprocess
import torch
import platform
from lib.download import download_all_data
from lib.pytorch_misc import set_seed

# Decide which pytorch version is available
try:
    from torchvision.ops import roi_align
    TORCH12 = True
    def tensor_item(x):
        return x.item()
    NO_GRAD = torch.no_grad
except:
    raise NotImplementedError('Pytorch version %s is no longer supported in this code' % str(torch.__version__))

try:
    import wandb
except Exception as e:
    print('wandb is not available: install it using pip install wandb', e)

MODES = ('sgdet', 'sgcls', 'predcls')

BOX_SCALE = 1024  # Scale at which we have the boxes
IM_SCALE = 592      # Our images will be resized to this res without padding

REL_FG_FRACTION = 0.25
BATCHNORM_MOMENTUM = 0.01
FG_FRACTION = 0.25

# Proposal assignments
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0

ROIS_PER_IMG = 256

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self):
        """
        Defaults
        """
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        # print git commit to make sure the code is reproducible
        try:
            self.gitcommit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        except Exception as e:
            print(e)
            self.gitcommit = str(e)

        self.torch_version = torch.__version__
        self.cuda_version = torch.version.cuda
        self.hostname = platform.node()

        print("\n~~~~~~~~ Hyperparameters: ~~~~~~~")
        self.__dict__.update(self.args)
        for x, y in self.__dict__.items():
            if x in ['args', 'parser']:
                continue
            print("{} : {}".format(x, y))

        self.steps = list(map(int, self.steps.split('_')))
        assert self.val_size >= 0, self.val_size
        assert self.num_gpus == 1, ('this code does not support multiple gpus')

        if self.split == 'gqa':
            assert self.rels_per_img == 1024, '1024 rels should be used for GQA'

        if self.split != 'stanford':
            assert self.backbone == 'resnet50', (
                'Do not use VG pretrained detector on other splits since the train set might overlap with the test set')

        if self.save_dir == '':
            self.save_dir = None
        if self.save_dir is not None and not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if self.test_bias:
            assert self.use_bias, 'use_bias must be specified in this case '

        if self.perturb:
            assert self.gan, ('GAN must be used in case of perturbations', self.gan)
            assert self.L > 0 and self.L <= 1, ('perturbation intensity must be > 0 and <= 1', self.L)

        self.ganlosses = self.ganlosses.split('_')
        if self.gan:
            assert len(self.ganlosses) > 0, ('at least one GAN loss must be specified to train GAN', self.ganlosses)

        # Set seed everywhere
        set_seed(self.seed)

        if not os.path.exists(self.data):
            if len(self.data) == 0:
                raise ValueError("data must be a valid path")
            os.mkdir(self.data)

        if not self.data_exists():
            download_all_data(self.data, gqa=self.split == 'gqa', vg=self.split != 'gqa')

        # Weights and Biases (WANDB) logging tool (optional)
        try:
            if self.wandb is None or len(self.wandb) == 0:
                raise ValueError('project name must be specified if you want to use wandb')

            import wandb
            stop_keys = ['args', 'parser']
            wandb.init(name=self.name,
                       dir=self.wandb_dir,
                       project=self.wandb,
                       config=dict(filter(lambda k: k[0] not in stop_keys, self.__dict__.items())),
                       resume=False)

            def wandb_log(d, step, log_repeats=1, is_summary=False, prefix=''):
                for step__ in range(step, step + log_repeats):  # to fix a wandb issue of not syncing the last few values with the server
                    try:
                        for key, value in d.items():
                            wandb.log({prefix + key: value}, step=step__)
                            if is_summary:
                                wandb.run.summary[prefix + key] = value
                    except Exception as e:
                        print('error logging with wandb:', e)  # e.g. in case the disk is full

            self.wandb_log = wandb_log

        except Exception as e:
            print('\nwarning: Logging using Weights and Biases will not be used:', e)
            self.wandb_log = None


    def data_exists(self):
        return (self.split != 'gqa' and
                os.path.exists(os.path.join(self.data, 'VG', 'VG_100K')) and
                os.path.exists(os.path.join(self.data, 'VG', 'stanford_filtered'))) or \
                (self.split == 'gqa' and
                 os.path.exists(os.path.join(self.data, 'GQA', 'train_balanced_questions.json')))


    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')

        parser.add_argument('-data', dest='data', help='path where Visual Genome and GQA are located', type=str, default='./data')
        parser.add_argument('-split', dest='split', type=str, default='stanford', choices=['stanford', 'vte', 'gqa'])

        parser.add_argument('-ckpt', dest='ckpt', help='Filename to load from', type=str, default='')
        parser.add_argument('-save_dir', dest='save_dir', help='Directory to save things to, such as checkpoints/save', default=None, type=str)
        parser.add_argument('-notest', dest='notest', help='do not evaluate on the test set after training', action='store_true')
        parser.add_argument('-save_scores', dest='save_scores', help='save test predictions', action='store_true')


        parser.add_argument('-ngpu', dest='num_gpus', help='cuantos GPUs tienes', type=int, default=1)
        parser.add_argument('-nwork', dest='num_workers', help='num processes to use as workers', type=int, default=2)
        parser.add_argument('-seed', dest='seed', type=int, default=111, help='random seed for model parameters and others')
        parser.add_argument('-device', dest='device', help='cpu/cuda device to use (cpu might be useful for debugging)', type=str, default='cuda')


        # Main learning args
        parser.add_argument('-lr', dest='lr', help='learning rate', type=float, default=1e-3)
        parser.add_argument('-lr_decay', dest='lr_decay', help='learning rate decay factor', type=float, default=0.1)
        parser.add_argument('-steps', dest='steps', help='the epochs after which decay the learning rate', type=str, default='15')
        parser.add_argument('-nepoch', dest='num_epochs', help='Number of epochs to train the model for', type=int, default=20)
        parser.add_argument('-b', dest='batch_size', help='batch size per GPU',type=int, default=6)
        parser.add_argument('-val_size', dest='val_size', help='val size to use (if 0 we wont use val)', type=int, default=5000)
        parser.add_argument('-l2', dest='l2', help='weight decay', type=float, default=1e-4)
        parser.add_argument('-clip', dest='clip', help='gradients will be clipped to have norm less than this', type=float, default=5.0)
        parser.add_argument('-m', dest='mode', help='mode \in {sgdet, sgcls, predcls}', type=str, default='sgcls', choices=['sgdet', 'sgcls', 'predcls'])
        parser.add_argument('-use_bias', dest='use_bias',  action='store_true', help='Use Frequency Bias')
        parser.add_argument('-test_bias', dest='test_bias', action='store_true', help='Use only Frequency Bias')
        parser.add_argument('-edge_model', dest='edge_model', type=str, default='motifs',
                            choices=['motifs', 'raw_boxes'], help='which box features use for edges')
        parser.add_argument('-pred_weight', dest='pred_weight', help='weight for predicate softmax scores in the evaluation mode (denoted as x in the paper)', type=float, default=0)

        # SGG losses
        parser.add_argument('-loss', dest='loss', type=str, default='baseline', choices=['baseline', 'dnorm', 'dnorm-fgbg'],
                            help='type of loss for SG prediction')
        parser.add_argument('-gamma', dest='gamma', type=float, default=1.0,
                            help='weight for the relationship loss; '
                                 'in the paper lambda and gamma have the same effect, so we keep only one of them here')
        parser.add_argument('-alpha', dest='alpha', type=float, default=1.0, help='weight for the foreground edges')
        parser.add_argument('-beta', dest='beta', type=float, default=1.0, help='weight for the background edges')
        parser.add_argument('-rels_per_img', dest='rels_per_img', type=int, default=1024,
                            help='the maximum number of edges per image sampled during training')

        # Object detector
        parser.add_argument('-backbone', dest='backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'],
                            help='backbone used to detect objects and extract object/edge features')

        # Dataset
        parser.add_argument('-min_graph_size', dest='min_graph_size', type=int, default=-1,
                            help='min number of nodes used during training')
        parser.add_argument('-max_graph_size', dest='max_graph_size', type=int, default=-1,
                            help='max number of nodes used during training')
        parser.add_argument('-exclude_left_right', dest='exclude_left_right', help='exclude left/right relationships (for GQA)', action='store_true')

        # Extra args
        parser.add_argument('-p', dest='print_interval', help='print during training', type=int, default=100)
        parser.add_argument('-wandb', dest='wandb', type=str, default=None,
                            help='the name of the weights and biases project (empty to avoid wandb)')
        parser.add_argument('-wandb_dir', dest='wandb_dir',
                            help='directory for wandb logging (can take a lot of space)', type=str, default='./')
        parser.add_argument('-name', dest='name', help='name of the experiment', type=str, default=None)
        parser.add_argument('-debug', dest='debug', action='store_true')

        # GAN args
        parser.add_argument('-gan', dest='gan', action='store_true', help='train with GAN')
        parser.add_argument('-ganlosses', dest='ganlosses', type=str, default='D_G_rec')
        parser.add_argument('-lrG', dest='lrG', type=float, default=0.0001, help='for GAN with Adam optimizer')
        parser.add_argument('-lrD', dest='lrD', type=float, default=0.0004, help='for GAN with Adam optimizer')
        parser.add_argument('-ganw', dest='ganw', type=float, default=5.0, help='weight for the GAN loss terms')
        parser.add_argument('-vis_cond', dest='vis_cond', type=str, default=None, help='condition the GAN on precomputed visual features')
        parser.add_argument('-attachG', dest='attachG', action='store_true', help='update G on the reconstruction error')
        parser.add_argument('-init_embed', dest='init_embed', action='store_true',
                            help='initialize conditioning embeddings with the GloVe ones')
        parser.add_argument('-largeD', dest='largeD', action='store_true', help='use a larger discriminator')
        parser.add_argument('-beta1', dest='beta1', type=float, default=0, help='Adam hyperparameter for GAN, good values for GANs are 0 and 0.5')
        parser.add_argument('-beta2', dest='beta2', type=float, default=0.9, help='Adam hyperparameter for GAN, good values for GANs are 0.9 and 0.999')

        # Perturbation args
        parser.add_argument('-perturb', dest='perturb', type=str, default=None, choices=['rand', 'neigh', 'graphn'],
                            help='perturbation method, None denotes no perturbations')
        parser.add_argument('-L', dest='L', type=float, default=0.2,
                            help='perturbation intensity, 0 denotes no perturbations')
        parser.add_argument('-topk', dest='topk', type=int, default=5,
                            help='number of semantic neighbors to sample from')
        parser.add_argument('-graphn_a', dest='graphn_a', type=int, default=2,
                            help='alpha parameter of GraphN controlling rare vs frequent compositions')
        parser.add_argument('-uniform', dest='uniform', action='store_true', help='sample nodes uniformly instead of by node degree')
        parser.add_argument('-degree_smoothing', dest='degree_smoothing', type=float, default=1.0, help='flatten the degree distribution '
                                                                                                        '(ignored if uniform=True)')
        return parser
