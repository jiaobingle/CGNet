# CGNet
## Introduction
We first design a Context Guided (CG) block by considering the inherent characteristic of semantic segmentation. CG Block aggregates local feature, surrounding context feature and global context feature effectively and efficiently. Based on the CG block, we develop Context Guided Network (CGNet), which not only has a strong capacity of localization and recognition, but also has small memory footprint. Under a similar number of parameters, the proposed CGNet outperforms existing segmentation networks. Without any pre-processing or post-processing, the proposed approach achieves 63.8% mean IoU on Cityscapes test set with less than 0.5 M parameters, and has a frame-rate of 43 fps for 2048 × 1024 high-resolution image.

![image](img/CGNet.png)

## Results on Cityscapes test set
We train the proposed CGNet with only fine annotated data and submit our test results to the official evaluation server.
![image](img/results.png)

## Usage
1. Install PyTorch
  - The code is developed on python3.6 on Ubuntu 16.04. (GPU: Tesla K80; PyTorch: 0.5; Cuda: 8.0)
2. Clone the repository
   ```shell
   git clone https://github.com/wutianyiRosun/CGNet.git 
   cd CGNet
   ```
3. Dataset

  - Download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset and convert the dataset to [19 categories](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py). 
  
4. Training
  ```
    python train.py
  ```
5. Evaluation (on validation set)
 
  ```
    python eval.py
  ```
6. Testing (on test set)
  ```
    python test.py
  ```
