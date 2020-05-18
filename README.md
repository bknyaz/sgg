# Scene Graph Generation

| Object Detections |  Ground truth Scene Graph |  Generated Scene Graph
|:-------------------------:|:-------------------------:|:-------------------------:|
| <figure> <img src="figs/2320504_ours_zs_ours.png" height="200"></figure> |  <figure> <img src="figs/2320504_ours_zs_graph_gt.png" height="200"><figcaption></figcaption></figure> | <figure> <img src="figs/2320504_ours_zs_graph_ours.png" height="200"><figcaption></figcaption></figure> |

In this visualization, `woman sitting on rock` is a **zero-shot** triplet, which means that the combination of `woman`, `sitting on` and `rock` has never been observed during training. However, each of the object and predicate has been observed, but together with other objects and predicate. For example, `woman sitting on chair` has been observed and is not a zero-shot triplet.


This code accompanies our paper [Boris Knyazev, Harm de Vries, Cătălina Cangea, Graham W. Taylor, Aaron Courville, Eugene Belilovsky. "Graph Density-Aware Losses for Novel Compositions in Scene Graph Generation"](https://arxiv.org/search/cs?searchtype=author&query=Knyazev%2C+B)

To run our experiments we used amazing [Rowan Zellers' code for Neural Motifs](https://github.com/rowanz/neural-motifs). Its only problem is the difficult to run it in PyTorch > 0.3, which is often required on recent GPUs.

So, in this repo, I provide a cleaned-up version that can be run in PyTorch 1.2 or later. The code is based on Mask R-CNN built-in in recent PyTorch.
It should be possible to reproduce our GQA results using this code.

**This code does not require building or downloading anything in advance**. Training the Scene Graph Classification (SGCls) model is as easy as running this command:

`python main.py -data /path/to/VG`


**This repository is in progress, use at your own risk.**

## TODO

- [x] Message Passing with Mask R-CNN
- [ ] Automatically download all files required to run the code
- [ ] Obtain results on VG
- [ ] Obtain results on GQA
- [ ] Add the script to visualize scene graph generation used in the paper
