### 2024高级机器学习结课作业
### ——基于扩散模型的单目RGB图像三维人体姿态估计
陆雨晴 U202014942 人工智能与自动化学院



### 环境配置

代码在以下环境下开发和测试：

-   Python 3.8.2
-   PyTorch 1.7.1
-   CUDA 11.0

可以通过以下指令创建环境：

```bash
conda env create -f environment.yml
```

### 数据集

数据集基于[3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline)和[Video3D](https://github.com/facebookresearch/VideoPose3D) data。DiffPose提供了从上述数据集生成的GMM格式数据，下载链接在 [here](https://www.dropbox.com/sh/54lwxf9zq4lfzss/AABmpOzg31PrhxzcxmFQt3cYa?dl=0)，应该将下载的文件放入`./data`目录中。 DiffPose只是更改了Video3D数据的格式，以使其与基于GMM的DiffPose训练策略兼容，并且DiffPose数据集中的2D姿势值与它们相同。

## 基于帧的实验
### 评估基于帧的实验的预训练模型

提供预训练的扩散模型（以CPN检测到的2D姿势作为输入） [here](https://www.dropbox.com/sh/jhwz3ypyxtyrlzv/AABivC5oiiMdgPePxekzu6vga?dl=0). 要对其进行评估，请将其放入`./checkpoint`目录并运行：

```bash
CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py \
--config human36m_diffpose_uvxyz_cpn.yml --batch_size 1024 \
--model_pose_path checkpoints/gcn_xyz_cpn.pth \
--model_diff_path checkpoints/diffpose_uvxyz_cpn.pth \
--doc t_human36m_diffpose_uvxyz_cpn --exp exp --ni \
>exp/t_human36m_diffpose_uvxyz_cpn.out 2>&1 &
```

提供预训练的扩散模型（以真实的2D姿势作为输入）[here](https://www.dropbox.com/sh/jhwz3ypyxtyrlzv/AABivC5oiiMdgPePxekzu6vga?dl=0).要对其进行评估，请将其放入`./checkpoint`目录并运行：

```bash
CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py \
--config human36m_diffpose_uvxyz_gt.yml --batch_size 1024 \
--model_pose_path checkpoints/gcn_xyz_gt.pth \
--model_diff_path checkpoints/diffpose_uvxyz_gt.pth \
--doc t_human36m_diffpose_uvxyz_gt --exp exp --ni \
>exp/t_human36m_diffpose_uvxyz_gt.out 2>&1 &
```

### 训练新模型

-   要从头开始训练一个模型（使用CPN检测到的2D姿势作为输入），运行：

```bash
CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py --train \
--config human36m_diffpose_uvxyz_cpn.yml --batch_size 1024 \
--model_pose_path checkpoints/gcn_xyz_cpn.pth \
--doc human36m_diffpose_uvxyz_cpn --exp exp --ni \
>exp/human36m_diffpose_uvxyz_cpn.out 2>&1 &
```

-   要从头开始训练一个模型（使用真实的2D姿势作为输入），运行：
```bash
CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py --train \
--config human36m_diffpose_uvxyz_gt.yml --batch_size 1024 \
--model_pose_path checkpoints/gcn_xyz_gt.pth \
--doc human36m_diffpose_uvxyz_gt --exp exp --ni \
>exp/human36m_diffpose_uvxyz_gt.out 2>&1 &
```


## 引用

部分代码借鉴 [DiffPose](https://github.com/GONGJIA0208/Diffpose),[DDIM](https://github.com/ermongroup/ddim), [VideoPose3D](https://github.com/facebookresearch/VideoPose3D), [Graformer](https://github.com/Graformer/GraFormer), [MixSTE](https://github.com/JinluZhang1126/MixSTE) 和 [PoseFormer](https://github.com/zczcwh/PoseFormer). 感谢作者们发布了这些代码。


## 演示软件的使用
### 第一阶段：OpenPose的安装部署
OpenPose是由美国卡耐基梅隆大学开发，可用于人体、人脸、人手等关键点估计的首个基于深度学习的实时多人二维姿态估计应用。关于OpenPose的安装部署，已经有许多优秀的教程，可以参考[OpenPose官方文档](https://github.com/CMU-Perceptual-Computing-Lab/openpose)进行安装部署。也可以参考网友整理的[OpenPose安装部署教程](https://blog.csdn.net/qq_22841387/article/details/136930379)进行安装部署。

### 第二阶段：3D姿态估计
请在完成OpenPose的安装部署且加载完成权重文件后，运行：
```bash
python hpe_api.py
```