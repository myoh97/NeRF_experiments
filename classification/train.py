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

def train(args):
    device = torch.device('cuda:0')
    
    bs = args.bs
    n_iter = args.iter
    
    train_dataset = NeRFDataset(train=True, pop_layer=args.pop_layer)
    val_dataset = NeRFDataset(train=False, pop_layer=args.pop_layer)
    
    len_w = train_dataset.len_w
    len_b = train_dataset.len_b
    
    if args.model == 'base' : model = Baseline(len_w = len_w, len_b = len_b).to(device)
    elif args.model == 'light' : model = Baseline_light().to(device)
    elif args.model == 'light_drop' : model = Baseline_light_drop().to(device)

    save_dir = f"results/{args.exp}/{args.model}_bs{bs}_iter{n_iter}_pop{args.pop_layer}"
    
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=32)
    val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=32)

    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    best_val = 0.
    
    start = time.time()
    
    # epoch
    for iter in range(n_iter):
        # train
        model.train()
        for idx, (weight, bias, target) in enumerate(train_dataloader):
            weight = weight.to(device).requires_grad_()
            bias = bias.to(device).requires_grad_()
            target = target.to(device).reshape(-1)
            
            if args.normalize: 
                weight, bias = normalize(weight, bias)
                
            pred = model(weight, bias)
            loss = loss_func(pred, target) * args.w_loss + args.reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(pred, 1)
            acc = torch.sum(predicted==target) / target.shape[0]
        
        # validation
        model.eval()
        for idx, (weight, bias, target) in enumerate(val_dataloader):
            weight = weight.to(device)
            bias = bias.to(device)
            target = target.to(device).reshape(-1)
            
            pred = model(weight, bias)
        
            prob = F.softmax(pred, dim=-1)
            _, predicted = torch.max(pred, 1)
            
            acc_v = torch.sum(predicted==target) / target.shape[0]
            val_loss = loss_func(pred, target)            
            
            if iter % 100 == 0:
                show_predicted_result(predicted, target, prob)

        if acc_v > best_val:
            best_val = acc_v.item()
            print(f"Best acc : {best_val:.4f}")
            save_model(model, optimizer, save_dir)
            
        print(f"[{iter+1}/{n_iter}] Train Acc : {acc:.4f}, Train Loss : {loss:.4f}, Val Acc : {acc_v:.4f}, Val Loss : {val_loss:.4f}")

    end = time.time()
    print(f"{end - start:.5f} sec")
    
    test(model, save_dir, val_dataloader, device)
    
def show_predicted_result(predicted, target, prob):
    print("================================")
    print("prediction\t target\n")
    for p, t, pr in zip(predicted, target, prob):
        print(f"{class_dict[p.item()]}({pr[p]:.2f})\t{class_dict[t.item()]}")
    print("================================")
    
def normalize(weight, bias):
    weight -= weight.min(1, keepdim=True)[0]
    weight /= weight.max(1, keepdim=True)[0]
    bias -= bias.min(1, keepdim=True)[0]
    bias /= bias.max(1, keepdim=True)[0]

    return weight, bias

def save_model(model, optimizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir, f"best_ckpt.tar")
    
    torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }, save_name)

def test(model, save_dir, val_dataloader, device):
    print("Test on best checkpoint")
    model.eval()
    
    model.load_state_dict(os.path.join(save_dir, 'best_ckpt.tar'))
    
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
    
    f = open(os.path.join(save_dir, 'result.json'), 'w')
    f.writelines([f"Accuracy : {acc_v:.4f}", "\n", "================================", "\n"])
    f.writelines("\n".join([f"{class_dict[p.item()]}({pr[p]:.2f})\t{class_dict[t.item()]}" for p, t, pr in zip(predicted, target, prob)]))
    f.close()
    
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pop_layer', default=None)
    parser.add_argument('--iter', default=500, type=int) #
    parser.add_argument('--bs', default=362, type=int) #
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
