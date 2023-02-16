from nmodel import *
from dataset import NeRFDataset

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class_dict = {
    0:"chair",
    1:"drums",
    2:"ficus",
    3:"hotdog",
    4:"lego",
    5:"materials",
    6:"mic",
    7:"ship"
    }


def test(args):
    device = torch.device('cuda:0')
    
    bs = args.bs
    n_iter = args.iter
    
    if args.model == 'fc' : model = FCLayer(batchsize=bs).to(device)
    elif args.model =='fc_drop': model = FCLayer_dropout(batchsize=bs).to(device)
    elif args.model =='fc_light': model = FCLayer_light(batchsize=bs).to(device)
    elif args.model =='fc_heavy': model = FCLayer_heavy(batchsize=bs).to(device)
    elif args.model =='fc_bn': model = FCLayer_BN(batchsize=bs).to(device)
    elif args.model =='fc_in': model = FCLayer_IN(batchsize=bs).to(device)
    elif args.model =='fc_bn_drop': model = FCLayer_BN(batchsize=bs).to(device)
    elif args.model =='fc_in_drop': model = FCLayer_IN_dropout(batchsize=bs).to(device)

    val_dataset = NeRFDataset(train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=288, shuffle=False, num_workers=0)
    ckpt = torch.load(args.ckpt)
    
    model.load_state_dict(ckpt['state_dict'])
    
    start = time.time()

    model.eval()
    for idx, (weight, bias, target) in enumerate(val_dataloader):
        weight = weight.to(device)
        bias = bias.to(device)
        target = target.to(device).reshape(-1)
        pred = model(weight, bias)
    
        prob = F.softmax(pred, dim=-1)
        max_idx, predicted = torch.max(pred, 1)
        acc_v = torch.sum(predicted==target) / target.shape[0]
        
    
        print("================================")
        print("prediction\t target\n")
        for p, t, pr, i in zip(predicted, target, prob, max_idx):
            print(f"{class_dict[p.item()]}({pr[p]:.2f})\t{class_dict[t.item()]}")
        print("================================")
        
    print(f"Accuracy : {acc_v:.4f}")

    end = time.time()
    print(f"{end - start:.5f} sec")
    
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--iter', default=1000, type=int) #
    parser.add_argument('--bs', default=288, type=int) #
    parser.add_argument('--model', default='fc', choices=['fc', 'fc_drop', 'fc_bn', 'fc_bn_drop', 'fc_in', 'fc_in_drop', 'fc_light', 'fc_heavy'])
    parser.add_argument('--w_loss', default=1.0, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--exp', type=str, default='exp')
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()
    
    print(args)
    return args
    
def main():
    args = parse_args()
    test(args)

if __name__ == '__main__':
    main()