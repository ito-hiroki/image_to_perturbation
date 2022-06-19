# Overview
This is the official implementation of the paper "Image to Perturbation: An Image Transformation Network for Generating Visually Protected Images for Privacy-Preserving Deep Neural Networks."

# classification model

## train classification model
```
python train_classification_model.py --model resnet20 --dataset cifar10
```

## test classification model
```
python test_classification_model.py --model resnet20 --dataset cifar10
```

# transformation model
The transformation model is used for generating visually protected images for privacy-preserving deep neural networks.

## train transformation model
```
python train_transformation_model.py --model unet --classifier resnet20 --dataset cifar10 --p_ratio 0.01
```

## test transformation model
```
python test_transformation_model.py --model unet --classifier resnet20 --dataset cifar10 --p_ratio 0.01
```

# generate visually protected images
```
python generate_protected_dataset.py --model unet --dataset cifar10 --classifier resnet20 --p_ratio 0.01 --original
```

# inverse transformation model
The inverse transformation model is used for restoring original images from visually protected images.

## train inverse transformation model
```
python train_inverse_transformation_model.py --model unet --dataset ../dataset/protect/trans/cifar10/resnet20/0-01/ --ckpt_name inverse_unet_0-01_resnet20_cifar10.pth
```


