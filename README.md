# Learning Cross-Representation Affinity Consistency for Sparsely Supervised Biomedical Instance Segmentation
**Accepted by ICCV-2023**

Xiaoyu Liu, Wei Huang, Zhiwei Xiong*, Shenglong Zhou, Yueyi Zhang, Xuejin Chen, Zheng-jun Zha, and Feng Wu 

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




## Training stage

Take the training on the AC3 dataset as an example.

### 1. train CPSN

```shell
python pre_training.py -c=pretraining_snemi3d
```

### 2. train CRAC

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

```shell
 python inference.py -c=seg_snemi3d_d5_1024_u200 -mn=seg_ac3_d5_1024_u200_WS -id=seg_ac3_d5_1024_u200_WS -m=snemi3d-ac3
```



## Contact

If you have any problem with the released code and dataset, please contact me by email (liuxyu@mail.ustc.edu.cn).


