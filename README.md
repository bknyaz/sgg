# Scene Graph Generation

| Object Detections |  Ground truth Scene Graph |  Generated Scene Graph
|:-------------------------:|:-------------------------:|:-------------------------:|
| <figure> <img src="figs/2320504_ours_zs_ours.png" height="200"></figure> |  <figure> <img src="figs/2320504_ours_zs_graph_gt.png" height="200"><figcaption></figcaption></figure> | <figure> <img src="figs/2320504_ours_zs_graph_ours.png" height="200"><figcaption></figcaption></figure> |

In this visualization, `woman sitting on rock` is a **zero-shot** triplet, which means that the combination of `woman`, `sitting on` and `rock` has never been observed during training. However, each of the object and predicate has been observed, but together with other objects and predicate. For example, `woman sitting on chair` has been observed and is not a zero-shot triplet. Making correct predictions for zero-shots is very challenging, so in our paper we address this problem and improve zero-shot as well as few-shot results.


This code accompanies our paper [Boris Knyazev, Harm de Vries, Cătălina Cangea, Graham W. Taylor, Aaron Courville, Eugene Belilovsky. "Graph Density-Aware Losses for Novel Compositions in Scene Graph Generation"](https://arxiv.org/search/cs?searchtype=author&query=Knyazev%2C+B)


To run our experiments we used amazing [Rowan Zellers' code for Neural Motifs](https://github.com/rowanz/neural-motifs). Its only problem is the difficult to be run in PyTorch > 0.3, making it hard to use it on some recent GPUs.

So, in this repo, I provide a cleaned-up version that can be run in PyTorch 1.2 or later. The code is based on Mask R-CNN built-in in recent PyTorch.
It should be possible to reproduce our GQA results using this code.

**This code does not require building or manually downloading anything in advance**. Training the Scene Graph Classification (SGCls) model with our loss on Visual Genome is as easy as running this command:

`python main.py -data data_path -loss dnorm`

The script will automatically download all data and create the following directories (make sure you have at lease 30Gb in `data_path`):

```
data_path
│   VG
│   │   VG_100K
│   │   ...
│
└───GQA
│   │   sceneGraphs
|   |   ...
```

To run it on GQA, use:

`python main.py -data data_path -loss dnorm -split gqa -lr 0.002`

Checkpoints and predictions will be saved locally in `./results`. This can be changed by the `-save_dir` flag.

**This repository is still in progress, please report any issues.**

## Requirements

- Python > 3.5
- PyTorch >= 1.2
- Other standard libraries

Should be enough to install these libraries (in addition to PyTorch):
```
conda install -c anaconda h5py cython dill pandas
conda install -c conda-forge pycocotools tqdm
```

## TODO

- [x] Message Passing with Mask R-CNN
- [x] Automatically download all files required to run the code
- [x] Obtain SGCls/PredCls results on VG
- [ ] Obtain SGCls/PredCls results on GQA
- [ ] Obtain SGGen results on GQA
- [ ] Add the script to visualize scene graph generation used in the paper


## VG Results

Results here are obtained using Mask R-CNN with ResNet-50 as a backbone, while in the paper we used Faster R-CNN with VGG16 as a backbone, hence the difference. See full details in the paper. Pretraining Mask R-CNN  on VG should help to improve results.

| Loss | Detector |  SGCls-R@100 |  SGCls-R_ZS@100 | PredCls-R@50 | PredCls-R_ZS@50
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|
| Baseline, this repo | Mask R-CNN (ResNet-50) pretrained on COCO | 47.1 | 7.8 | 74.5 | 23.5 |
| D-norm (ours), this repo | Mask R-CNN (ResNet-50) pretrained on COCO |47.4 | 9.0 | 75.4 | 27.3
| D-norm (ours), paper | Faster R-CNN (VGG16) pretrained on VG |48.6 | 9.1 | 78.2 | 28.4

## GQA Results

## Scene Graph Visualizations

## Citation

Please use this citation if you want to cite our paper:

```
@article{knyazev2020graphdensity,
  title={Graph Density-Aware Losses for Novel Compositions in Scene Graph Generation},
  author={Knyazev, Boris and de Vries, Harm and Cangea, Cătălina and Taylor, Graham W and Courville, Aaron and Belilovsky, Eugene},
  journal={arXiv preprint arXiv:2005.08230},
  year={2020}
}
```
