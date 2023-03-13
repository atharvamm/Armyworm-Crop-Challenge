# Crop Fall Armyworm Recognition

> Fall Armyworm is a pest found in Africa that destroys maize produce resulting in losses to farmers. The goal was to design a convolutional neural network(CNN) to recognize maize affected by the pest from the unaffected ones.
I implemented a CNN similar to [AlexNet](https://en.wikipedia.org/wiki/AlexNet) to classify and recognize the infected maize crops.

> A text representation of the model.
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 96, 54, 54]          34,944
              GELU-2           [-1, 96, 54, 54]               0
         MaxPool2d-3           [-1, 96, 26, 26]               0
            Conv2d-4          [-1, 256, 26, 26]         614,656
              GELU-5          [-1, 256, 26, 26]               0
         MaxPool2d-6          [-1, 256, 14, 14]               0
            Conv2d-7          [-1, 512, 14, 14]       1,180,160
              GELU-8          [-1, 512, 14, 14]               0
            Conv2d-9         [-1, 1024, 14, 14]       4,719,616
             GELU-10         [-1, 1024, 14, 14]               0
           Conv2d-11         [-1, 2048, 14, 14]      18,876,416
             GELU-12         [-1, 2048, 14, 14]               0
        MaxPool2d-13           [-1, 2048, 7, 7]               0
           Conv2d-14           [-1, 4096, 7, 7]      75,501,568
             GELU-15           [-1, 4096, 7, 7]               0
AdaptiveAvgPool2d-16           [-1, 4096, 1, 1]               0
          Flatten-17                 [-1, 4096]               0
           Linear-18                 [-1, 1024]       4,195,328
      BatchNorm1d-19                 [-1, 1024]           2,048
          Dropout-20                 [-1, 1024]               0
             GELU-21                 [-1, 1024]               0
           Linear-22                  [-1, 256]         262,400
      BatchNorm1d-23                  [-1, 256]             512
          Dropout-24                  [-1, 256]               0
             GELU-25                  [-1, 256]               0
           Linear-26                    [-1, 2]             514
================================================================
Total params: 105,388,162
Trainable params: 105,388,162
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 22.44
Params size (MB): 402.02
Estimated Total Size (MB): 425.04
----------------------------------------------------------------
```
---
## Link to the dataset
[Link](https://zindi.africa/competitions/makerere-fall-armyworm-crop-challenge/data)