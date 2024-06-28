from __future__ import absolute_import, division

import numpy as np
import torch

from common.utils import wrap
from common.quaternion import qrot, qinverse


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

def add_noise_me(data,index,x_or_y):

        # 定义不同关节上的误差
    ex=np.array([
        [-0.0003,0.0081],
        [-0.0005,0.01],
        [-0.0003,0.0113],
        [-0.0005,0.0141],
        [-0.0003,0.0094],
        [-0.0011,0.0117],
        [-0.0008,0.0144],
        [-0.0003,0.0098],
        [-0.0007,0.0087],
        [-0.0005,0.0082],
        [-0.0009,0.0076],
        [-0.001,0.0091],
        [-0.0011,0.0139],
        [-0.0001,0.0153],
        [-0.0003,0.009],
        [-0.0004,0.0131],
        [-0.0011,0.0150]
    ])

    ey=np.array([
        [-0.0005,0.0053],
        [0.0019,0.0076],
        [0.0029,0.0093],
        [0.0024,0.0102],
        [-0.0052,0.0078],
        [-0.0016,0.0095],
        [-0.0017,0.011],
        [-0.0017,0.0056],
        [-0.0031,0.0065],
        [-0.0011,0.0048],
        [-0.0008,0.0038],
        [-0.0024,0.006],
        [-0.0053,0.0109],
        [-0.0019,0.0198],
        [-0.0011,0.0063],
        [-0.0033,0.011],
        [0.0003,0.0197]
    ])
    # 原代码
    # dataa=[length,17,3,2]
    # length = data.shape[0]
    # for j in range(0,17): #17个关节
    #     random_sample = np.random.normal(ex[j][0], ex[j][1], length)
    #     temp=data[:,j,0,0]
    #     temp=temp+random_sample
    #     data[:,j,0,0]=temp

    # for j in range(0,17): #17个关节
    #     random_sample = np.random.normal(ey[j][0], ey[j][1], length)
    #     temp=data[:,j,0,1]
    #     temp=temp+random_sample
    #     data[:,j,0,1]=temp

    # 0331改
    
    # length = data.shape[0]
    if x_or_y==1:
        data[1]= data[1]+np.random.normal(ex[index][0], ex[index][1], 1)
    elif x_or_y==2:
        data[2]= data[2]+np.random.normal(ey[index][0], ey[index][1], 1)
    return data 
    # random_sample_y = np.random.normal(ey[index][0], ey[index][1], 1)
    # data[index,1]=data[index,1]+random_sample_x
    # data[index,2]=data[index,2]+random_sample_y
    # return data 


def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2


def world_to_camera(X, R, t):
    Rt = wrap(qinverse, False, R)  # Invert rotation
    return wrap(qrot, False, np.tile(Rt, X.shape[:-1] + (1,)), X - t)  # Rotate and translate


def camera_to_world(X, R, t):
    return wrap(qrot, False, np.tile(R, X.shape[:-1] + (1,)), X) + t


def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=len(r2.shape) - 1), dim=len(r2.shape) - 1,
                           keepdim=True)
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)

    XXX = XX * (radial + tan) + p * r2

    return f * XXX + c


def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)

    return f * XX + c
