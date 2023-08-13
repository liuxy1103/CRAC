# Learning Cross-Representation Affinity Consistency for Sparsely Supervised Biomedical Instance Segmentation
**Accepted by ICCV-2023**

Xiaoyu Liu, Wei Huang, Zhiwei Xiong*, Shenglong Zhou, Yueyi Zhang, Xuejin Chen, Zhengjun Zha, and Feng Wu 

University of Science and Technology of China (USTC), Hefei, China

Institute of Artificial Intelligence, Hefei Comprehensive National Science Center, Hefei, China

*Corresponding Author

## Abstract
In this paper, we propose a sparsely supervised biomedical instance segmentation framework via  cross-representation affinity consistency regularization. Specifically, we adopt two individual networks to enforce the perturbation consistency between an explicit affinity map and an implicit affinity map to capture both feature-level instance discrimination and pixel-level instance boundary structure. We then select the highly confident region of each affinity map as the pseudo label to supervise the other one for affinity consistency learning. To obtain the highly confident region, we propose a pseudo-label noise filtering scheme by integrating two entropy-based decision strategies. Extensive experiments on four biomedical datasets with sparse instance annotations show the state-of-the-art performance of our proposed framework. For the first time, we demonstrate the superiority of sparse instance-level supervision on 3D volumetric datasets, compared to common semi-supervision under the same annotation cost.


## Enviromenent

This code was tested with Pytorch 1.0.1 (later versions may work), CUDA 9.0, Python 3.7.4 and Ubuntu 16.04. 

If you have a [Docker](https://www.docker.com/) environment, we strongly recommend you to pull our image as follows：

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/renwu527/auto-emseg:v3.1
```


## Dataset

| Datasets   | Sizes                        | Resolutions | Species | Download (Processed) |
| ---------- | ---------------------------- | ----------- | ----------- | ----------- |
| [AC3/AC4 ](https://software.rc.fas.harvard.edu/lichtman/vast/AC3AC4Package.zip)   | 1024x1024x256, 1024x1024x100 | 6x6x30 nm^3 | Mouse | [BaiduYun](https://pan.baidu.com/s/1sSTkh7g9tccb_uZOvySQqQ) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1JAdoKchlWrHnbTXvnFn6pWWwx6VIiMH3?usp=sharing) |
| [CREMI](https://cremi.org/)      | 1250x1250x125 (x3)           | 4x4x40 nm^3 | Drosophila | [BaiduYun](https://pan.baidu.com/s/1q-irVm5aoSXL5eQiqyYs1w) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1JAdoKchlWrHnbTXvnFn6pWWwx6VIiMH3?usp=sharing) |
| [Kasthuri15](https://lichtman.rc.fas.harvard.edu/vast/Thousands_6nm_spec_lossless.vsv) | 10747x12895x1850             | 6x6x30 nm^3 | Mouse | [BaiduYun](https://pan.baidu.com/s/136Eml2gBHYIklVPP0MI_kQ) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1JAdoKchlWrHnbTXvnFn6pWWwx6VIiMH3?usp=sharing) |



## Training stage

Take the training on the AC3 dataset as an example.

### 1. Pre-training

```shell
python pre_training.py -c=pretraining_snemi3d
```

### 2. Consistency learning

Weight Sharing (WS)

```shell
python main.py -c=seg_snemi3d_d5_u200
```

EMA

```shell
python main_ema.py -c=seg_snemi3d_d5_1024_u200_ema
```



## Validation stage

Take the validation on the AC3 dataset as an example.





## Contact

If you have any problem with the released code and dataset, please contact me by email (liuxyu@mail.ustc.edu.cn).


