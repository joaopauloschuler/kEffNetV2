# Grouped Pointwise Convolution With Flexible Number of Channels and Filters per Group

## Abstract
In image classification with Deep Convolutional Neural Networks (DCNN), the number of parameters in pointwise convolutions rapidly grows due to the multiplication of the number of filters by the number of input channels that come from the previous layer. In previous work, we showed a subnetwork that replaces pointwise convolutions with significantly fewer parameters and fewer floating-point computations while maintaining the learning capacity. Our subnetwork utilizes grouped pointwise convolutions, in which each group processes a fraction of the input channels. In the present work, we refine the previous algorithm so that groups are allowed to have a number of filters to cope with non-divisible numbers of input channels, output channels, and groups, in which case our previous method overlooked and did not replace the original pointwise convolutions. Thus, the new method further reduces the number of floating-point computations (11\%) and trainable parameters (10\%) achieved by the previous method. We tested our optimization on an EfficientNet-B0 as a baseline architecture and made classification tests on the CIFAR-10, Colorectal Cancer Histology, and Malaria datasets. For each dataset, our optimization achieves a saving of 76\%, 89\%, and 91\% of the number of trainable parameters of EfficientNet-B0, while keeping its test classification accuracy.

## Raw Results
Raw results and source code will be stored in this repository once the paper is accepted for publication.

## Citing this Paper
TBC
