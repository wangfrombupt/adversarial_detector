# New Adversarial Image Detection Based on Sentiment Analysis

This project is for the paper "[New Adversarial Image Detection Based on Sentiment Analysis]()".Some codes are from [Mahalanobis detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector),

## Preliminaries
It is tested under Ubuntu Linux 18.04 and Python 3.7 environment, and requries Pytorch package to be installed:

* [Pytorch v1.10](http://pytorch.org/): Only GPU version is available.

We use the following two libraries to generate adversarial examples.
* [adversarial robustness toolbox(ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
* [Foolbox](https://github.com/bethgelab/foolbox)

## Training Classifier
we train ResNet-34 on CIFAR-10, CIFAR-100 and SVHN.
```
# model: ResNet-34, dataset: CIFAR-10, gpu: 0
python train.py --dataset cifar10 --net_type resnet --gpu 0
```

## Detecting Adversarial Samples

### 0. Generate adversarial samples:
```
# model: ResNet, dataset: CIFAR-10, adversarial attack: FGSM, gpu: 0
python adv_generate.py --dataset cifar10 --net_type resnet --adv_type FGSM --gpu 0
```

### 1. Train detector:
```
# model: ResNet, dataset: CIFAR-10, adversarial attack: FGSM, gpu: 0
python train_detector.py --dataset cifar10 --adv_type FGSM --gpu 0
```

## Evaluating on PGD White Box Attack
### 0. Generate original adversarial samples:
```
# model: ResNet, dataset: CIFAR-10, adversarial attack: PGD-20, gpu: 0
python adv_generate.py --dataset cifar10 --net_type resnet --adv_type PGD-20 --gpu 0
```

### 1. Train original detector:
```
# model: ResNet, dataset: CIFAR-10, adversarial attack: PGD-20, gpu: 0
python train_detector.py --dataset cifar10 --net_type resnet --adv_type PGD-20 --gpu 0
```

### 2. Adversarial training on White Box Attack samples:
```
cd white_box
# dataset: CIFAR-10, model: ResNet, original attack: PGD-20, gpu: 0
python adv_train.py --dataset cifar10 --adv_type PGD-20 --gpu 0
```

### 3. Test the robustness
```
# dataset: CIFAR-10, model: ResNet, original attack: PGD-20, gpu: 0
# white box attack: pgd-iter-20, pgd-combine-20, resume_epoch: 2
python test_robust.py --dataset cifar10 --adv_type PGD-20 --gpu 0 --resume 2
```

