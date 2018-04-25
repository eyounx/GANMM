# GANMM
GANMM code for the paper
> Yang Yu, Wen-Ji Zhou. **Mixture of GANs for clustering**. In: Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI'18), Stockholm, Sweden.

## Requirement
- python 3.5
- argparse
- pickle
- tensorflow(tested with GPU version) == 1.0.0
- numpy == 1.12.1
- sklearn == 0.18.1

## Files

- GANMM.py  implements the algorihtm
- main.py   is the demo that uses GANMM to cluster some data sets as in the paper
- Data      contains data sets
- nets      network structure
- tflib     tensorflow components (modified from https://github.com/igul222/improved_wgan_training)

## Run Demo
- To run experiment on mnist data set, just running:
```bash
# mnist raw data
python main.py mnist

# mnist preprocessed by stacked auto-encoder
python main.py sae_mnist
```
GPU version of TensorFlow is recommended for mnist data set. Tensorflow (CPU) may not support `data_format="NCHW"` in Conv2DBackpropFilter operate.

- Two UCI-dataset:
```bash
# Image Segmentation data set
python main.py seg

# Artificial Characters data set
python main.py chara
```
- On different data scale:
```bash
python main.py seg --scale 0.5
```
