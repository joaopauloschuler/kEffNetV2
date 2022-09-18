# An Enhanced Scheme for Reducing the Complexity of Pointwise Convolutions in CNNs for Image Classification Based on Interleaved Grouped Filters without Divisibility Constraints

This repository contains the source code for the paper [An Enhanced Scheme for Reducing the Complexity of Pointwise Convolutions in CNNs for Image Classification Based on Interleaved Grouped Filters without Divisibility Constraints (PDF)](https://www.researchgate.net/publication/363413038_An_Enhanced_Scheme_for_Reducing_the_Complexity_of_Pointwise_Convolutions_in_CNNs_for_Image_Classification_Based_on_Interleaved_Grouped_Filters_without_Divisibility_Constraints).

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

## Further Parameter and Floating-point Computation Savings
The following papers also deal about parameters and floating-point computation savings:
- [Grouped Pointwise Convolutions Reduce Parameters in Convolutional Neural Networks.](https://www.researchgate.net/publication/360226228_Grouped_Pointwise_Convolutions_Reduce_Parameters_in_Convolutional_Neural_Networks)
- [Reliable Deep Learning Plant Leaf Disease Classification Based on Light-Chroma Separated Branches.](https://www.researchgate.net/publication/355215213_Reliable_Deep_Learning_Plant_Leaf_Disease_Classification_Based_on_Light-Chroma_Separated_Branches)
- [Color-aware two-branch DCNN for efficient plant disease classification.](https://www.researchgate.net/publication/361511874_Color-Aware_Two-Branch_DCNN_for_Efficient_Plant_Disease_Classification)

## Creating Optimized Models - Beyond the Paper
Beyond what is shown on the paper, you can create kDenseNet, kInception V3 and kMobileNet optimized models as follows.

### kDenseNet-BC L100 12ch
In DenseNets, you can define the minimum number of channels per group for transitions (`kTypeTransition`) and for blocks (`kTypeBlock`):
```
model = cai.densenet.ksimple_densenet([32, 32, 3], 
  blocks=16, 
  growth_rate=12, bottleneck=48, compression=0.5,
  l2_decay=0,
  kTypeTransition=cai.layers.D6v3_12ch(),
  kTypeBlock=cai.layers.D6v3_12ch(), 
  num_classes=10,
  dropout_rate=0.0,
  activation=keras.activations.swish,
  has_interleave_at_transition=True)
```

### kInception V3 32ch
The example below should work for most use cases:
```
model = cai.inception_v3.two_path_inception_v3(
  include_top=True,
  weights=None,
  input_shape=(224, 224, 3),
  pooling=None,
  classes=num_classes,
  two_paths_partial_first_block=0,
  two_paths_first_block=False,
  two_paths_second_block=False,
  deep_two_paths=False,
  kType=cai.layers.D6v3_32ch())
```
Some of the parameters such as `two_paths_partial_first_block`, `two_paths_first_block` and `two_paths_second_block` are related to the papaer [Reliable Deep Learning Plant Leaf Disease Classification Based on Light-Chroma Separated Branches](https://github.com/joaopauloschuler/two-path-noise-lab-plant-disease).

### kMobileNet 32ch

The example below creates a basic with optimized pointwise convolutions:

```
model = cai.mobilenet.kMobileNet(
  include_top=True,
  weights=None,
  input_shape=(224, 224, 3),
  pooling=None,
  classes=10,
  kType=cai.layers.D6v3_32ch())
```

### kMobileNet V3 32ch
The example below creates a MobileNet V3 with optimized pointwise convolutions:
```
model = cai.mobilenet_v3.kMobileNetV3Large(
  input_shape=(224, 224, 3),
  alpha=1.0,
  minimalistic=False,
  include_top=True,
  input_tensor=None,
  classes=10,
  pooling=None,
  dropout_rate=0.2,
  kType=cai.layers.D6v3_32ch())
```

## Give this Project a Star
This project is an open source project. If you like what you see, please give it a star on github.

## Citing this Paper
```
@Article{e24091264,
AUTHOR = {Schwarz Schuler, Joao Paulo and Also, Santiago Romani and Puig, Domenec and Rashwan, Hatem and Abdel-Nasser, Mohamed},
TITLE = {An Enhanced Scheme for Reducing the Complexity of Pointwise Convolutions in CNNs for Image Classification Based on Interleaved Grouped Filters without Divisibility Constraints},
JOURNAL = {Entropy},
VOLUME = {24},
YEAR = {2022},
NUMBER = {9},
ARTICLE-NUMBER = {1264},
URL = {https://www.mdpi.com/1099-4300/24/9/1264},
ISSN = {1099-4300},
DOI = {10.3390/e24091264}
}
```
