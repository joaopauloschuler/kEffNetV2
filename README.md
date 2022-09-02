# An Enhanced Scheme for Reducing the Complexity of Pointwise Convolutions in CNNs for Image Classification Based on Interleaved Grouped Filters without Divisibility Constraints

## Abstract
In image classification with Deep Convolutional Neural Networks (DCNNs), the number of parameters in pointwise convolutions rapidly grows due to the multiplication of the number of filters by the number of input channels that come from the previous layer. Existing studies demonstrated that a subnetwork can replace pointwise convolutional layers with significantly fewer parameters and fewer floating-point computations, while maintaining the learning capacity. In this paper, we propose an improved scheme for reducing the complexity of pointwise convolutions in DCNNs for image classification based on interleaved grouped filters without divisibility constraints. The proposed scheme utilizes grouped pointwise convolutions, in which each group processes a fraction of the input channels. It requires a number of channels per group as a hyperparameter (Ch). The subnetwork of the proposed scheme contains two consecutive convolutional layers K and L, connected by an interleaving layer in the middle, and summed at the end. The number of groups of filters and filters per group for layers K and L is determined by exact divisions of the original number of input channels and filters by (Ch). If the divisions were not exact, the original layer could not be substituted. In this paper, we refine the previous algorithm so that input channels are replicated and groups can have different numbers of filters to cope with non exact divisibility situations. Thus, the proposed scheme further reduces the number of floating-point computations (11\%) and trainable parameters (10\%) achieved by the previous method. We tested our optimization on an EfficientNet-B0 as a baseline architecture and made classification tests on the CIFAR-10, Colorectal Cancer Histology, and Malaria datasets. For each dataset, our optimization achieves a saving of 76\%, 89\%, and 91\% of the number of trainable parameters of EfficientNet-B0, while keeping its test classification accuracy.

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
