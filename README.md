# Adversarial-Dropout

Tensorflow implementation for the results in the paper "Adversarial Dropout for Supervised and Semi-supervised Learning" (https://arxiv.org/abs/1707.03631)

This implementation is based on the Code from Takeru Miyato's repository at https://github.com/takerum/vat_tf
(Thank for Takeru Miyato's Work)


## Dependency

This work was tested with Tensorflow 1.4.1, CUDA 8.0, python 2.7 

## Preparation of dataset

CIFAR10 for semi-supervised learning

```python cifar10.py```

## Semi-Supervised Learning on CIFAR10

With Virtual Adversarial Dropout with KL loss

```python train.py --dataset=cifar10 --data_dir=dataset/cifar10/ --log_dir=log/cifar10_semisup_VAdD-KL --method=VAdD --num_epochs=300 --mean_only_bn=True --aug_trans=True --aug_flip=True --sigma=0.15 --lamb_max=1.0 --delta=0.05```

## Implementation of Experiments in Paper

Check the branch, "experiments---TF1.1.0". 

