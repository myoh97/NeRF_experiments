import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import numpy as np

color_dict = {
    "rgb":0,
    "rbg":1,
    "grb":2,
    "gbr":3,
    "brg":4,
    "bgr":5
    }

nerf_class = {
    "sofa":0,
    "chair":1,
    "table":2
}

class NeRFDataset(Dataset):
    def __init__(self, train=True, pop_layer = None):
        self.param_list = ['020000.tar']
        self.train_path = "/root/dataset/NeRF/nerf_parameter_3d/train/*/*/"
        self.val_path = "/root/dataset/NeRF/nerf_parameter_3d/val/*/*/"
        self.phase = 'train' if train==True else 'validation'
        
        if train == True:
            self.dir_list = glob.glob(self.train_path)
        else:
            self.dir_list = glob.glob(self.val_path)

        self.weights = []
        self.biases = []
        self.targets = []
        
        # pop_layer = 0
        for d in self.dir_list: # path
            for p in self.param_list: # weight
                ckpt = dict(torch.load(d+p, map_location=torch.device('cpu'))['network_fine_state_dict'])
                ckpt_ = dict()
                
                for key in ckpt.keys():
                    layer = key.split('.')[0]
                    num = key.split('.')[1]
                    types = key.split('.')[-1]
                    
                    ###! layer ablation to figure out which layer is important
                    # weight or bias
                    if pop_layer in ['bias', 'weight'] and types == pop_layer:
                        continue
                    # views, feature, rgb, alpha
                    if type(pop_layer) == str and layer.split('_')[0] == pop_layer:
                        continue
                    # pts_linears 0~7
                    if layer != 'views_linears' and num == str(pop_layer):
                        continue                       
                    #########################################################!
                    
                    ckpt_[key] = ckpt[key].numpy().flatten()
                        
                w = np.concatenate([ckpt_[key] for key in ckpt_.keys() if key.split('.')[-1] == 'weight'])
                b = np.concatenate([ckpt_[key] for key in ckpt_.keys() if key.split('.')[-1] == 'bias'])
                t = nerf_class[d.split('/')[-3]]
                
                self.weights.append(w)
                self.biases.append(b)
                self.targets.append(t)
                
        print(f"Loaded {len(self.weights)} trained models for {self.phase}")
        
        self.len_w = len(w)
        self.len_b = len(b)
        
    def __len__(self):
        return len(self.weights)

    def __getitem__(self, idx):
        
        w = torch.as_tensor(self.weights[idx], dtype=torch.float32)
        b = torch.as_tensor(self.biases[idx], dtype=torch.float32)
        t = torch.as_tensor(self.targets[idx], dtype=torch.int64)
        
        return w, b, t