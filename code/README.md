# Regularized Mask Tuning: Uncovering Hidden Knowledge in Pre-trained Vision-Language Models
Official implementation of ['Regularized Mask Tuning: Uncovering Hidden Knowledge in Pre-trained Vision-Language Models](https://arxiv.org/abs/2307.15049)'.

## How to Install

This code is built on top of [CoOP](https://github.com/KaiyangZhou/CoOp). So you need to install the environment following CoOP first. After that, run pip install -r requirements.txt to install a few more packages.

Follow [DATASET.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to install 11 datasets referring to CoOp.


## How to Run

The running scripts are provided in `scripts/masktuning/`, which allow you to reproduce the results on the ICCV'23 paper.


### Few-shot Classification
This corresponds to the experiments in Section 4.2, i.e., Fig 4.

You will need `scripts/masktuning/train.sh` for training. The script has two input arguments, i.e., `DATASET` and `GDR_LAMBDA`. `DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. `GDR_LAMBDA` is the parameter in Eq 7, which is set to be is set to 0.3 for datasets except for ImageNet, SUN397, and Food101 in 16-shot experiments. And l is set to 1.0 in other experiments. 

For evaluation, you will need `scripts/masktuning/eval.sh`. The script has one input argument, i.e., `DATASET`.

Below we provide an example on how to train and evaluate the model on ImageNet and Caltech101.

```bash
# train
bash scripts/masktuning/train.sh imagenet 3e-1
bash scripts/masktuning/train.sh caltech101 10e-1

# eval
bash scripts/masktuning/eval.sh imagenet 
bash scripts/masktuning/eval.sh caltech101
```

### Generalization From Base to New Classes

This corresponds to the experiments in Section 4.2, i.e., Table 1.

You will need `scripts/masktuning/base2new_train.sh` and `scripts/masktuning/base2new_eval.sh` for training and evaluation. Both scripts have one input argument, i.e., `DATASET`. 

Below we provide an example on how to train and evaluate the model on ImageNet.

```bash
# train
bash scripts/masktuning/base2new_train.sh imagenet 

# eval
bash scripts/masktuning/base2new_eval.sh imagenet
```