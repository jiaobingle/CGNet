# CGNet
## Introduction
We first design a Context Guided (CG) block by considering the inherent characteristic of semantic segmentation. CG Block aggregates local feature, surrounding context feature and global context feature effectively and efficiently. Based on the CG block, we develop Context Guided Network (CGNet), which not only has a strong capacity of localization and recognition, but also has small memory footprint. Under a similar number of parameters, the proposed CGNet outperforms existing segmentation networks. Without any pre-processing or post-processing, the proposed approach achieves 63.8% mean IoU on Cityscapes test set with less than 0.5 M parameters, and has a frame-rate of 43 fps for 2048 × 1024 high-resolution image.

![image][img/CGNet.png]

## Results on Cityscapes test set
We train the proposed CGNet with only fine annotated data and submit our test results to the official evaluation server.
![image][img/results.png]
