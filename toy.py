import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

import os
import os.path as osp
import sys
import torch
import torch.nn.functional as F
# from utils import *

prefix = '/root/dataset/NeRF/nerf_parameter'
classes = ['lego', 'chair', 'drums', 'ficus', 'hotdog', 'materials', 'mic', 'ship']
iters = ['050000', '060000', '070000', '080000', '090000', '100000']
colors = ['bgr', 'brg', 'gbr', 'grb', 'rbg', 'rgb']
phases = ['train', 'val']

mod = sys.modules[__name__] 

weights = []

# Load weights : "{class}_{phase}_{color}_{iter}"
for phase in phases:
    for cls in classes:
        for color in colors:
            for iter in iters:
                setattr(mod, f'{cls}_{phase}_{color}_{iter}', torch.load(osp.join(prefix, phase, cls, color, f'{iter}.tar'))['network_fine_state_dict'])
                print(f"Loaded {cls}_{phase}_{color}_{iter}")
                weights.append(eval(f"{cls}_{phase}_{color}_{iter}"))
                tmp = eval(f"{cls}_{phase}_{color}_{iter}"); break;

keys = list(weights[0].keys())

print(lego_train_rgb_050000)


