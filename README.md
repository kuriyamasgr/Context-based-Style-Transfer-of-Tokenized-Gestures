# Context-based-Style-Transfer-of-Tokenized-Gestures
The code and dataset for the paper of repository name published in Computer Graphics Forum (presented at ACM SIGGRAPH/EG Symposium on Computer Animation 2020)

### Prerequisite

- python 3.9
- torch 1.12.0
- pytorch_lightning 1.5.8
- glob
- numpy
- copy
- fastdtw
- scipy

### Training of autoencoder

% python autoencoder.py

### Training of gesture-style transformer

% python gsxf.py

### Evaluation of gesture-style transformer

% python evaluator.py

### Dataset preparation

Download BVH from https://bit.ly/42SrQmr
after uncompressing, put it under a dataset directory
