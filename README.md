# GANMM
GANMM for paper **Mixture of GANs for Clustering** (IJCAI 2018)

## Requirement
- python 3.5
- tensorflow
- numpy
- sklearn
- argparse
- pickle

## Run Demo
To run experiment on mnist data set, just running:
```bash
# mnist raw data
python main.py mnist

# mnist preprocessed by stacked auto-encoder
python main.py sae_mnist
```

Two UCI-dataset:
```bash
# Image Segmentation data set
python main.py seg

# Artificial Characters data set
python main.py chara
```
On different data scale:
```bash
python main.py seg --scale 0.5
```
