# Grouped Pointwise Convolution With Flexible Number of Channels and Filters per Group

## Abstract
In image classification with Deep Convolutional Neural Networks (DCNN), the number of parameters in pointwise convolutions rapidly grows due to the multiplication of the number of filters by the number of input channels that come from the previous layer. In previous work, we showed a subnetwork that replaces pointwise convolutions with significantly fewer parameters and fewer floating-point computations while maintaining the learning capacity. Our subnetwork utilizes grouped pointwise convolutions, in which each group processes a fraction of the input channels. In the present work, we refine the previous algorithm so that groups are allowed to have a number of filters to cope with non-divisible numbers of input channels, output channels, and groups, in which case our previous method overlooked and did not replace the original pointwise convolutions. Thus, the new method further reduces the number of floating-point computations (11\%) and trainable parameters (10\%) achieved by the previous method. We tested our optimization on an EfficientNet-B0 as a baseline architecture and made classification tests on the CIFAR-10, Colorectal Cancer Histology, and Malaria datasets. For each dataset, our optimization achieves a saving of 76\%, 89\%, and 91\% of the number of trainable parameters of EfficientNet-B0, while keeping its test classification accuracy.

## Quick Start on Your Own Web Browser
You can quickly give a go to our optimized kEffNet via [Google Colab](https://colab.research.google.com/) on your own browser:
* [kEffNet v2.](https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/kEffNet_v2.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaopauloschuler/k-neural-api/blob/master/examples/jupyter/kEffNet_v2.ipynb)

## The Raw Results Folder
If you just need an easy to use example, you can try the Colab example above. Otherwise, you can look at all raw experiment files used for the paper on the [raw](https://github.com/joaopauloschuler/kEffNetV2/tree/main/raw) folder.

## Creating Optimized Models
You can create kEffNet V2 following the example below. The parameter `kType` defines the minimum number of channels per group. For example, for 16 channels per group, you should use `cai.layers.D6v3_16ch()`. For 32 channels, you should use `cai.layers.D6v3_32ch()`.

All examples in this section require importing the [K-CAI Neural API](https://github.com/joaopauloschuler/k-neural-api).

### kEffNet V2
The example below shows the creation of a kEffNet-B0 16ch:
```
model = cai.efficientnet.kEfficientNetB0(
  include_top=True,
  input_shape=(224, 224, 3),
  classes=10,
  kType=cai.layers.D6v3_16ch())
```
For loading small images such as CIFAR-10's 32x32 images, you can skip the first strides with the `skip_stride_cnt` parameter as shown in the following example:
```
model = cai.efficientnet.kEfficientNetB0(
  include_top=True,
  skip_stride_cnt=3,
  input_shape=(32, 32, 3),
  classes=10,
  kType=cai.layers.D6v3_16ch())
```
Other [kEffNet variants](https://github.com/joaopauloschuler/k-neural-api/blob/master/cai/efficientnet.py) up to B7 are also available.

## Give this Project a Star
This project is an open source project. If you like what you see, please give it a star on github.

## Citing this Paper
TBC
