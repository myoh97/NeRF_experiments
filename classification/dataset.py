import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob

class_dict = {
    "chair":0,
    "drums":1,
    "ficus":2,
    "hotdog":3,
    "lego":4,
    "materials":5,
    "mic":6,
    "ship":7
    }

color_dict = {
    "rgb":0,
    "rbg":1,
    "grb":2,
    "gbr":3,
    "brg":4,
    "bgr":5
    }

class NeRFDataset(Dataset):
    def __init__(self, train=True):
        self.data_list = []

        self.rgb_list = ["rgb/","rbg/","grb/","gbr/","brg/","bgr/"]
        self.param_list = ['050000.tar','060000.tar','070000.tar','080000.tar','090000.tar','100000.tar']
        self.train_path = "/root/dataset/NeRF/nerf_parameter/val/*/"
        self.val_path = "/root/dataset/NeRF/nerf_parameter/train/*/"
        self.train = 'train' if train==True else 'validation'
        
        if train == True:
            self.dir_list = glob.glob(self.train_path)
        else:
            self.dir_list = glob.glob(self.val_path)

        self.weights = []
        self.biases = []
        self.targets = []
        
        for d in self.dir_list:
            for r in self.rgb_list:
                for p in self.param_list:
                    ckpt = dict(torch.load(d+r+p, map_location=torch.device('cpu'))['network_fine_state_dict'])
                    w = torch.cat([ckpt[key].flatten() for key in ckpt.keys() if key.split('.')[-1] == 'weight'])
                    b = torch.cat([ckpt[key].flatten() for key in ckpt.keys() if key.split('.')[-1] == 'bias'])
                    t = torch.Tensor([class_dict[d.split('/')[-2]]]).type(torch.LongTensor)
                    self.weights.append(w)
                    self.biases.append(b)
                    self.targets.append(t)
        print(f"Loaded {len(self.weights)} trained models for {self.train}")
        
    def __len__(self):
        return len(self.weights)

    def target(self, name):
        cls, color = name.split('/')[-3:-1]
        
        # target = torch.Tensor([class_dict[cls]*len(color_dict)+color_dict[color]]).type(torch.LongTensor)
        # # print(cls, color, target)
        # assert target < 48
        # return target
    
        if cls=="chair":
            return torch.Tensor([0]).type(torch.LongTensor)
        elif cls=="drums":
            return torch.Tensor([1]).type(torch.LongTensor)
        elif cls=="ficus":
            return torch.Tensor([2]).type(torch.LongTensor)
        elif cls=="hotdog":
            return torch.Tensor([3]).type(torch.LongTensor)
        elif cls=="lego":
            return torch.Tensor([4]).type(torch.LongTensor)
        elif cls=="materials":
            return torch.Tensor([5]).type(torch.LongTensor)
        elif cls=="mic":
            return torch.Tensor([6]).type(torch.LongTensor)
        elif cls=="ship":
            return torch.Tensor([7]).type(torch.LongTensor)

    def __getitem__(self, idx):
        return self.weights[idx], self.biases[idx], self.targets[idx]
        # x = self.data_list[idx]
        # weight = torch.load(x)#, map_location='cpu'
        # param = dict(weight["network_fine_state_dict"])
        # t = self.target(x)
        # return param, t