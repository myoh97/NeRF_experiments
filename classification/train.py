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

def train(args):
    device = torch.device('cuda:0')
    
    bs = args.bs
    n_iter = args.iter
    
    if args.model == 'base' : model = Baseline().to(device)
    elif args.model == 'light' : model = Baseline_light().to(device)
    elif args.model == 'light_drop' : model = Baseline_light_drop().to(device)

    save_dir = f"results/{args.exp}/{args.model}_bs{bs}_iter{n_iter}"
    
    train_dataset = NeRFDataset(train=True)
    val_dataset = NeRFDataset(train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=288, shuffle=False, num_workers=0)

    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    
    train_list = []
    val_list = []
    val_acc = []
    train_acc = []

    best_val = 0.
    
    start = time.time()
    
    for iter in range(n_iter):
        model.train()
        for idx, (weight, bias, target) in enumerate(train_dataloader):
            weight = weight.to(device).requires_grad_()
            bias = bias.to(device).requires_grad_()
            target = target.to(device).reshape(-1)
            
            if args.normalize:
                weight -= weight.min(1, keepdim=True)[0]
                weight /= weight.max(1, keepdim=True)[0]
                bias -= bias.min(1, keepdim=True)[0]
                bias /= bias.max(1, keepdim=True)[0]
                
            pred = model(weight, bias)
            loss = loss_func(pred, target) * args.w_loss + args.reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(pred, 1)
            acc = torch.sum(predicted==target) / target.shape[0]

        model.eval()
        for idx, (weight, bias, target) in enumerate(val_dataloader):
            weight = weight.to(device)
            bias = bias.to(device)
            target = target.to(device).reshape(-1)
            pred = model(weight, bias)
        
            prob = F.softmax(pred, dim=-1)
            _, predicted = torch.max(pred, 1)
            acc_v = torch.sum(predicted==target) / target.shape[0]
            
            if iter % 100 == 0:
                print("================================")
                print("prediction\t target\n")
                for p, t, pr in zip(predicted, target, prob):
                    print(f"{class_dict[p.item()]}({pr[p]:.2f})\t{class_dict[t.item()]}")
                print("================================")
                
            val_loss = loss_func(pred, target)
        
        if acc_v > best_val:
            best_val = acc_v.item()
            
            os.makedirs(save_dir, exist_ok=True)
            save_name = os.path.join(save_dir, f"iter{iter}_{best_val:.4f}.tar")
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_name)
            
            print(f"Best acc : {best_val:.4f}, saved {save_name}")
        print(f"[{iter+1}/{n_iter}] Train Acc : {acc:.4f}, Train Loss : {loss:.4f}, Val Acc : {acc_v:.4f}, Val Loss : {val_loss:.4f}")

    end = time.time()
    print(f"{end - start:.5f} sec")
    
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--iter', default=1000, type=int) #
    parser.add_argument('--bs', default=288, type=int) #
    parser.add_argument('--model', default='base', type=str)
    parser.add_argument('--w_loss', default=1.0, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--exp', type=str, default='exp')
    parser.add_argument('--reg', type=float, default=0.0)
    parser.add_argument('--normalize', action='store_true', default = False)
    args = parser.parse_args()
    
    print(args)
    return args
    
def main():
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main()
