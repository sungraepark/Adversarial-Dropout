# Adversarial-Dropout

Tensorflow implementation for the results in the paper "Adversarial Dropout for Supervised and Semi-supervised Learning" (https://arxiv.org/abs/1707.03631)

This implementation is based on the Code from Takeru Miyato's repository at (https://github.com/takerum/vat_tf)
(Thank for Takeru Miyato's job)

## Preparation of dataset

CIFAR10 for supervised learning

```python cifar10_sup.py```

SVHN for supervised learning

```python svhn_sup.py```

CIFAR10 for semi-supervised learning

```python cifar10_semisup.py```

SVHN for semi-supervised learning

```python svhn_semisup.py```

## Supervised Learning on CIFAR10

With Supervised Adversarial Dropout

```python train_sup.py --dataset=cifar10 --data_dir=dataset/cifar10/ --log_dir=log/cifar10_sup_SAdD --method=SAdD --num_epochs=300 --mean_only_bn=True --aug_trans=True --aug_flip=True --sigma=0.15 --lamb_max=1.0 --delta=0.05```

With Virtual Adversarial Dropout with KL loss

```python train_sup.py --dataset=cifar10 --data_dir=dataset/cifar10/ --log_dir=log/cifar10_sup_VAdD-KL --method=VAdD-KL --num_epochs=300 --mean_only_bn=True --aug_trans=True --aug_flip=True --sigma=0.15 --lamb_max=1.0 --delta=0.05```

With Virtual Adversarial Dropout with QE loss

```python train_sup.py --dataset=cifar10 --data_dir=dataset/cifar10/ --log_dir=log/cifar10_VAdD-QE --method=VAdD-QE --num_epochs=300 --mean_only_bn=True --aug_trans=True --aug_flip=True --sigma=0.15 --lamb_max=25.0 --delta=0.05```

### With Joint Learning with Virtual Adversarial Training + Virtual Adversarial Dropout

(KL loss)

```python train_sup.py --dataset=cifar10 --data_dir=dataset/cifar10/ --log_dir=log/cifar10_sup_VAT+VAdD-KL --method=VAT+VAdD-KL --mean_only_bn --num_epochs=300 --epsilon=8.0 --aug_trans=True --aug_flip=True --sigma=0.15 --lamb_max=1.0 --top_bn --delta=0.05```

(QE loss)

```python train_sup.py --dataset=cifar10 --data_dir=dataset/cifar10/ --log_dir=log/cifar10_sup_VAT+VAdD-QE --method=VAT+VAdD-QE --mean_only_bn  --num_epochs=300 --epsilon=8.0 --aug_trans=True --aug_flip=True --sigma=0.15 --lamb_max=25.0 --top_bn --delta=0.05```


## Supervised Learning on SVHN

With Supervised Adversarial Dropout

```python train_sup.py --dataset=svhn --data_dir=dataset/svhn/ --log_dir=log/svhn_sup_SAdD --method=SAdD --num_epochs=300 --aug_trans=True --lamb_max=1.0 --top_bn --delta=0.05```

With Virtual Adversarial Dropout with KL loss

```python train_sup.py --dataset=svhn --data_dir=dataset/svhn/ --log_dir=log/svhn_sup_VAdD-KL --method=VAdD-KL --num_epochs=300 --aug_trans=True --lamb_max=1.0 --top_bn --delta=0.05```

With Virtual Adversarial Dropout with QE loss

```python train_sup.py --dataset=svhn --data_dir=dataset/svhn/ --log_dir=log/svhn_sup_VAdD-QE --method=VAdD-QE --num_epochs=300 --aug_trans=True --lamb_max=10.0 --top_bn --delta=0.05```

### With Joint Learning with Virtual Adversarial Training + Virtual Adversarial Dropout

(KL loss)

```python train_sup.py --dataset=svhn --data_dir=dataset/svhn/ --log_dir=log/svhn_sup_VAT+VAdD-KL --method=VAT+VAdD-KL --num_epochs=300 --epsilon=3.5 --aug_trans=True --lamb_max=1.0 --top_bn --delta=0.05```

(QE loss)

```python train_sup.py --dataset=svhn --data_dir=dataset/svhn/ --log_dir=log/svhn_sup_VAT+VAdD-QE --method=VAT+VAdD-QE --num_epochs=300 --epsilon=3.5 --aug_trans=True --lamb_max=10.0 --top_bn --delta=0.05```


## Semi-Supervised Learning on CIFAR10

With Supervised Adversarial Dropout

```python train_semisup.py --dataset=cifar10 --data_dir=dataset/cifar10/ --log_dir=log/cifar10_semisup_SAdD --method=SAdD --num_epochs=300 --mean_only_bn=True --aug_trans=True --lamb_max=1.0 --sigma=0.15 --delta=0.05```

With Virtual Adversarial Dropout with KL loss

```python train_semisup.py --dataset=cifar10 --data_dir=dataset/cifar10/ --log_dir=log/cifar10_semisup_VAdD-KL --method=VAdD-KL --num_epochs=300 --mean_only_bn=True --aug_trans=True --aug_flip=True --sigma=0.15 --lamb_max=1.0 --delta=0.05```

With Virtual Adversarial Dropout with QE loss

```python train_semisup.py --dataset=cifar10 --data_dir=dataset/cifar10/ --log_dir=log/cifar10_semisup_VAdD-QE --method=VAdD-QE --num_epochs=300 --mean_only_bn=True --aug_trans=True --aug_flip=True --sigma=0.15 --lamb_max=25.0 --delta=0.05```

### With Joint Learning with Virtual Adversarial Training + Virtual Adversarial Dropout

(KL loss)

```python train_semisup.py --dataset=cifar10 --data_dir=dataset/cifar10/ --log_dir=log/cifar10_semisup_VAT+VAdD-KL --method=VAT+VAdD-KL --num_epochs=300 --mean_only_bn=True --epsilon=8.0 --aug_trans=True --aug_flip=True --sigma=0.15 --lamb_max=1.0 --delta=0.05```

(QE loss)

```python train_semisup.py --dataset=cifar10 --data_dir=dataset/cifar10/ --log_dir=log/cifar10_semisup_VAT+VAdD-QE --method=VAT+VAdD-QE --num_epochs=300 --mean_only_bn=True --epsilon=8.0 --aug_trans=True --aug_flip=True --sigma=0.15 --lamb_max=25.0 --delta=0.05```

## Semi-Supervised Learning on SVHN

With Supervised Adversarial Dropout

```python train_semisup.py --dataset=svhn --data_dir=dataset/svhn/ --log_dir=log/svhn_semisup_SAdD --method=SAdD --num_epochs=300 --aug_trans=True --lamb_max=1.0 --top_bn --delta=0.05```

With Virtual Adversarial Dropout with KL loss

```python train_semisup.py --dataset=svhn --data_dir=dataset/svhn/ --log_dir=log/svhn_semisup_VAdD-KL --method=VAdD-KL --num_epochs=300 --aug_trans=True --lamb_max=1.0 --top_bn --delta=0.05```

With Virtual Adversarial Dropout with QE loss

```python train_semisup.py --dataset=svhn --data_dir=dataset/svhn/ --log_dir=log/svhn_semisup_VAdD-QE --method=VAdD-QE --num_epochs=300 --aug_trans=True --lamb_max=25.0 --top_bn --delta=0.05```

### With Joint Learning with Virtual Adversarial Training + Virtual Adversarial Dropout

(KL loss)

```python train_semisup.py --dataset=svhn --data_dir=dataset/svhn/ --log_dir=log/svhn_semisup_VAT+VAdD-KL --method=VAT+VAdD-KL --num_epochs=300 --epsilon=3.5 --aug_trans=True --lamb_max=1.0 --top_bn --delta=0.05```

(QE loss)

```python train_semisup.py --dataset=svhn --data_dir=dataset/svhn/ --log_dir=log/svhn_semisup_VAT+VAdD-QE --method=VAT+VAdD-QE --num_epochs=300 --epsilon=3.5 --aug_trans=True --lamb_max=25.0 --top_bn --delta=0.05
```




