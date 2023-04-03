## Context-based-Style-Transfer-of-Tokenized-Gestures
The code and dataset for the paper of repository name published in Computer Graphics Forum (presented at ACM SIGGRAPH/EG Symposium on Computer Animation 2020) [[preprint]](https://bit.ly/40RbkkH)

### Prerequisite

- python 3.*
- torch
- pytorch_lightning
- glob
- numpy
- copy
- fastdtw
- scipy

### Training of autoencoder

% python nn/autoencoder.py

The log and checkpoints file will be saved at `````./pretrain/AE`````.

### Training of gesture-style transformer

% python nn/gsxf.py

The log and checkpoints file will be saved at `````./pretrain/GSXF`````.

### Evaluation of gesture-style transformer (Only CPU mode is available)

% python evaluator.py

### Dataset preparation

Download BVH from a [[zip file (43MB)]](https://bit.ly/3M1V24n),
and place its uncompressed folder under a dataset folder.

### Citation

If you use this work please cite.
```
@article {Kuriyama2022cgf,
journal = {Computer Graphics Forum},
title = {Context-based Style Transfer of Tokenized Gestures},
author = {Kuriyama, Shigeru and Mukai, Tomohiko and Taketomi, Takafumi and Mukasa, Tomoyuki},
year = {2022},
publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
ISSN = {1467-8659},
DOI = {10.1111/cgf.14645}
}
```

