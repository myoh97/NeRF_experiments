from nmodel import *
from dataset import NeRFDataset

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def train(args):
    device = torch.device('cuda:0')
    
    bs = args.bs
    n_iter = args.iter
    
    if args.model == 'fc' : model = FCLayer(batchsize=bs).to(device)
    elif args.model =='fc_drop': model = FCLayer_dropout(batchsize=bs).to(device)
    elif args.model =='fc_bn': model = FCLayer_BN(batchsize=bs).to(device)
    elif args.model =='fc_in': model = FCLayer_IN(batchsize=bs).to(device)
    elif args.model =='fc_bn_drop': model = FCLayer_BN(batchsize=bs).to(device)
    elif args.model =='fc_in_drop': model = FCLayer_IN_dropout(batchsize=bs).to(device)

    save_dir = f"results/{args.exp}/{args.model}_bs{bs}_iter{n_iter}"
    
    train_dataset = NeRFDataset(train=True)
    val_dataset = NeRFDataset(train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=64, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=64, pin_memory=True)

    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    
    train_list = []
    val_list = []
    val_acc = []
    train_acc = []

    best_val = 0.
    
    for iter in range(n_iter):
        model.train()
        for idx, (weight, bias, target) in enumerate(train_dataloader):
            weight = weight.to(device).requires_grad_()
            bias = bias.to(device).requires_grad_()
            target = target.to(device).reshape(-1)
            
            pred = model(weight, bias)
            loss = loss_func(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(pred, 1)
            acc = torch.sum(predicted==target) / target.shape[0]

            # train_list.append(loss.item())
            # train_acc.append(acc.item())

        model.eval()
        for idx, (weight, bias, target) in enumerate(val_dataloader):
            # for k in param.keys():
            #     param[k] = param[k].to(device)
            weight = weight.to(device)
            bias = bias.to(device)
            target = target.to(device).reshape(-1)
            pred = model(weight, bias)
            
            _, predicted = torch.max(pred, 1)
            acc_v = torch.sum(predicted==target) / target.shape[0]
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

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--iter', default=1000, type=int) #
    parser.add_argument('--bs', default=288, type=int) #
    parser.add_argument('--model', choices=['fc', 'fc_drop', 'fc_bn', 'fc_bn_drop', 'fc_in', 'fc_in_drop'])
    parser.add_argument('--exp', type=str, default='exp')
    args = parser.parse_args()
    
    print(args)
    return args
    
def main():
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main()
