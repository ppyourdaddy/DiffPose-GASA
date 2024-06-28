from __future__ import print_function, absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset
from functools import reduce
from .camera import add_noise_me


class PoseGenerator_gmm(Dataset):
    def __init__(self, poses_3d, poses_2d_gmm, actions, camerapara):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d_gmm = np.concatenate(poses_2d_gmm)
        self._actions = reduce(lambda x, y: x + y, actions)
        self._camerapara = np.concatenate(camerapara)
        self._kernel_n = self._poses_2d_gmm.shape[2]#?

        self._poses_3d[:,:,:] = self._poses_3d[:,:,:]-self._poses_3d[:,:1,:]

        assert self._poses_3d.shape[0] == self._poses_2d_gmm.shape[0] and self._poses_3d.shape[0] == len(self._actions)
        print('Generating {} poses...'.format(len(self._actions)))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d_gmm = self._poses_2d_gmm[index]
        out_action = self._actions[index] 
        out_camerapara = self._camerapara[index]

        # randomly select a kernel from gmm
        out_pose_2d_kernel = np.zeros([out_pose_2d_gmm.shape[0],out_pose_2d_gmm.shape[2]])#每个关节不同的分布
        for i in range(out_pose_2d_gmm.shape[0]):
            out_pose_2d_kernel[i] = out_pose_2d_gmm[i,np.random.choice(self._kernel_n, 1, p=out_pose_2d_gmm[i,:,0]).item()]
            # np.random.choice解释：
            # a=3;size=1;p是一个三个数的一维数组;
            # p始终为【1,0，0】；
            # 即out_pose_2d_kernel[i]=out_pose_2d_gmm[i,0];

            # 改 加噪
            # 每个关节有不同的分布，加噪时，每个关节的分布都不一样
            # 加噪的概率为0.3(对于xy,分开考虑概率)
            # if np.random.rand()<0.3:
            #     out_pose_2d_kernel[i]=add_noise_me(out_pose_2d_kernel[i],i,1)
            # if np.random.rand()<0.3:
            #     out_pose_2d_kernel[i]=add_noise_me(out_pose_2d_kernel[i],i,2)



        # generate uvxyz and uvxyz noise scale
        kernel_mean = out_pose_2d_kernel[:,1:3] # 各个关节的xy作平均
        # 改
        # kernel_variance = np.zeros((17, 2))
        kernel_variance = out_pose_2d_kernel[:,3:] # 最后两维作方差

        out_pose_uvxyz = np.concatenate((kernel_mean,out_pose_3d),axis=1)
        out_pose_noise_scale = np.concatenate((kernel_variance,np.ones(out_pose_3d.shape)),axis=1)

        out_pose_uvxyz = torch.from_numpy(out_pose_uvxyz).float()
        out_pose_noise_scale = torch.from_numpy(out_pose_noise_scale).float()
        out_pose_2d = torch.from_numpy(kernel_mean).float()
        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_camerapara = torch.from_numpy(out_camerapara).float()
        
        return out_pose_uvxyz, out_pose_noise_scale, out_pose_2d, out_pose_3d, out_action, out_camerapara

    def __len__(self):
        return len(self._actions)