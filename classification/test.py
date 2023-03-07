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
    0:"sofa",
    1:"chair",
    2:"table"
    }

def test(args):
    device = torch.device('cuda:0')
    
    bs = args.bs
    n_iter = args.iter
    
    val_dataset = NeRFDataset(train=False, pop_layer=args.pop_layer)
    val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0)
    ckpt = torch.load(args.ckpt)
    
    len_w = val_dataset.len_w
    len_b = val_dataset.len_b
    
    if args.model == 'base' : model = Baseline(len_w=len_w, len_b=len_b).to(device)
    save_path = os.path.dirname(args.ckpt)
    
    model.load_state_dict(ckpt['state_dict'])
    
    start = time.time()

    model.eval()
    for idx, (weight, bias, target) in enumerate(val_dataloader):
        weight = weight.to(device)
        bias = bias.to(device)
        target = target.to(device).reshape(-1)
        pred = model(weight, bias)
    
        prob = F.softmax(pred, dim=-1)
        _, predicted = torch.max(pred, 1)
        acc_v = torch.sum(predicted==target) / target.shape[0]
        
        print("================================")
        print("prediction\t target\n")
        for p, t, pr in zip(predicted, target, prob):
            print(f"{class_dict[p.item()]}({pr[p]:.2f})\t{class_dict[t.item()]}")
        print("================================")
        
    print(f"Accuracy : {acc_v:.4f}")

    end = time.time()
    
    f = open(os.path.join(save_path, 'result.json'), 'w')
    f.writelines([f"Accuracy : {acc_v:.4f}", "\n", "================================", "\n"])
    f.writelines("\n".join([f"{class_dict[p.item()]}({pr[p]:.2f})\t{class_dict[t.item()]}" for p, t, pr in zip(predicted, target, prob)]))
    f.close()
    
    print(f"{end - start:.5f} sec")
    
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pop_layer', type=int, default=None)
    parser.add_argument('--iter', default=1000, type=int) #
    parser.add_argument('--bs', default=288, type=int) #
    parser.add_argument('--model', default='base')
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