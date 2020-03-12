[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/NVlabs/DG-Net.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/NVlabs/DG-Net/context:python)

## Joint Discriminative and Generative Learning for Person Re-identification
![](NxN.jpg)

[[Project]](http://zdzheng.xyz/DG-Net/) [[Paper]](https://arxiv.org/abs/1904.07223) [[YouTube]](https://www.youtube.com/watch?v=ubCrEAIpQs4) [[Bilibili]](https://www.bilibili.com/video/av51439240) [[Poster]](http://zdzheng.xyz/images/DGNet_poster.pdf)
[[Supp]](http://jankautz.com/publications/JointReID_CVPR19_supp.pdf)

Joint Discriminative and Generative Learning for Person Re-identification, CVPR 2019 (Oral)<br>
[Zhedong Zheng](http://zdzheng.xyz/), [Xiaodong Yang](https://xiaodongyang.org/), [Zhiding Yu](https://chrisding.github.io/), [Liang Zheng](http://liangzheng.com.cn/), [Yi Yang](https://www.uts.edu.au/staff/yi.yang), [Jan Kautz](http://jankautz.com/) <br>

## Table of contents
* [News](#news)
* [Features](#features)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
    * [Installation](#installation)
    * [Dataset Preparation](#dataset-preparation)
    * [Testing](#testing)
    * [Training](#training)
* [DG-Market](#dg-market)
* [Tips](#tips)
* [Citation](#citation)
* [Related Work](#related-work)
* [License](#license)

## News
- 08/24/2019: We add the direct transfer learning results of DG-Net [here](https://github.com/NVlabs/DG-Net#person-re-id-evaluation).
- 08/01/2019: We add the support of multi-GPU training: `python train.py --config configs/latest.yaml  --gpu_ids 0,1`.

## Features
We have supported:
- Multi-GPU training (fp32)
- [APEX](https://github.com/NVIDIA/apex) to save GPU memory (fp16/fp32)
- Multi-query evaluation
- Random erasing
- Visualize training curves
- Generate all figures in the paper 

## Prerequisites

- Python 3.6
- GPU memory >= 15G (fp32)
- GPU memory >= 10G (fp16/fp32)
- NumPy
- PyTorch 1.0+
- [Optional] APEX (fp16/fp32)

## Getting Started
### Installation
- Install [PyTorch](http://pytorch.org/) 
- Install torchvision from the source:
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- [Optional] You may skip it. Install APEX from the source:
```
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```
- Clone this repo:
```bash
git clone https://github.com/NVlabs/DG-Net.git
cd DG-Net/
```

Our code is tested on PyTorch 1.0.0+ and torchvision 0.2.1+ .

### Dataset Preparation
Download the dataset [Market-1501](http://www.liangzheng.com.cn/Project/project_reid.html) [[Google Drive]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu Disk]](https://pan.baidu.com/s/1ntIi2Op)

Preparation: put the images with the same id in one folder. You may use 
```bash
python prepare-market.py          # for Market-1501
```
Note to modify the dataset path to your own path.

### Testing

#### Download the trained model
We provide our trained model. You may download it from [Google Drive](https://drive.google.com/open?id=1lL18FZX1uZMWKzaZOuPe3IuAdfUYyJKH) (or [Baidu Disk](https://pan.baidu.com/s/1503831XfW0y4g3PHir91yw) password: rqvf). You may download and move it to the `outputs`.
```
├── outputs/
│   ├── E0.5new_reid0.5_w30000
├── models
│   ├── best/                   
```
#### Person re-id evaluation
- Supervised learning

|   | Market-1501  | DukeMTMC-reID  | MSMT17  | CUHK03-NP |
|---|--------------|----------------|----------|-----------|
| Rank@1 | 94.8% | 86.6% | 77.2% | 65.6% |
| mAP    | 86.0% | 74.8% | 52.3% | 61.1% |


- Direct transfer learning     
To verify the generalizability of DG-Net, we train the model on dataset A and directly test the model on dataset B (with no adaptation). 
We denote the direct transfer learning protocol as `A→B`.

|   |Market→Duke|Duke→Market|Market→MSMT|MSMT→Market|Duke→MSMT|MSMT→Duke|
|---|----------------|----------------| -------------- |----------------| -------------- |----------------|
| Rank@1  | 42.62%   | 56.12%         | 17.11%         | 61.76%         | 20.59%         | 61.89%         |
| Rank@5  | 58.57%   | 72.18%         | 26.66%         | 77.67%         | 31.67%         | 75.81%         |
| Rank@10 | 64.63%   | 78.12%         | 31.62%         | 83.25%         | 37.04%         | 80.34%         |
| mAP     | 24.25%   | 26.83%         | 5.41%          | 33.62%         | 6.35%          | 40.69%         |


#### Image generation evaluation

Please check the `README.md` in the `./visual_tools`. 

You may use the `./visual_tools/test_folder.py` to generate lots of images and then do the evaluation. The only thing you need to modify is the data path in [SSIM](https://github.com/layumi/PerceptualSimilarity) and [FID](https://github.com/layumi/TTUR).

### Training

#### Train a teacher model
You may directly download our trained teacher model from [Google Drive](https://drive.google.com/open?id=1lL18FZX1uZMWKzaZOuPe3IuAdfUYyJKH) (or [Baidu Disk](https://pan.baidu.com/s/1503831XfW0y4g3PHir91yw) password: rqvf).
If you want to have it trained by yourself, please check the [person re-id baseline](https://github.com/layumi/Person_reID_baseline_pytorch) repository to train a teacher model, then copy and put it in the `./models`.
```
├── models/
│   ├── best/                   /* teacher model for Market-1501
│       ├── net_last.pth        /* model file
│       ├── ...
```

#### Train DG-Net 
1. Setup the yaml file. Check out `configs/latest.yaml`. Change the data_root field to the path of your prepared folder-based dataset, e.g. `../Market-1501/pytorch`.


2. Start training
```
python train.py --config configs/latest.yaml
```
Or train with low precision (fp16)
```
python train.py --config configs/latest-fp16.yaml
```
Intermediate image outputs and model binary files are saved in `outputs/latest`.

3. Check the loss log
```
 tensorboard --logdir logs/latest
```

## DG-Market
![](https://github.com/layumi/DG-Net/blob/gh-pages/index_files/DGMarket-logo.png)

We provide our generated images and make a large-scale synthetic dataset called DG-Market. This dataset is generated by our DG-Net and consists of 128,307 images (613MB), about 10 times larger than the training set of original Market-1501 (even much more can be generated with DG-Net). It can be used as a source of unlabeled training dataset for semi-supervised learning. You may download the dataset from [Google Drive](https://drive.google.com/file/d/126Gn90Tzpk3zWp2c7OBYPKc-ZjhptKDo/view?usp=sharing) (or [Baidu Disk](https://pan.baidu.com/s/1n4M6s-qvE08J8SOOWtWfgw) password: qxyh).  

|   |  DG-Market   | Market-1501 (training) |
|---|--------------|-------------|
| #identity| 	-   |  751        |
| #images| 128,307 |  12,936     |

## Tips
Note the format of camera id and number of cameras. For some datasets (e.g., MSMT17), there are more than 10 cameras. You need to modify the preparation and evaluation code to read the double-digit camera id. For some vehicle re-id datasets (e.g., VeRi) having different naming rules, you also need to modify the preparation and evaluation code.

## Citation
Please cite this paper if it helps your research:
```bibtex
@inproceedings{zheng2019joint,
  title={Joint discriminative and generative learning for person re-identification},
  author={Zheng, Zhedong and Yang, Xiaodong and Yu, Zhiding and Zheng, Liang and Yang, Yi and Kautz, Jan},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

## Related Work
Other GAN-based methods compared in the paper include [LSGAN](https://github.com/layumi/DCGAN-pytorch), [FDGAN](https://github.com/layumi/FD-GAN) and [PG2GAN](https://github.com/charliememory/Pose-Guided-Person-Image-Generation). We forked the code and made some changes for evaluatation, thank the authors for their great work. We would also like to thank to the great projects in [person re-id baseline](https://github.com/layumi/Person_reID_baseline_pytorch), [MUNIT](https://github.com/NVlabs/MUNIT) and [DRIT](https://github.com/HsinYingLee/DRIT).

## License
Copyright (C) 2019 NVIDIA Corporation. All rights reserved. Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**). The code is released for academic research use only. For commercial use, please contact [researchinquiries@nvidia.com](researchinquiries@nvidia.com).
