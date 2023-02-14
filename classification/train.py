from nmodel import *
from dataset import NeRFDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

bs = 288

train_dataset = NeRFDataset(train=True)
val_dataset = NeRFDataset(train=False)

train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

model = FCLayer_woBN(batchsize=bs).cuda()
loss_func = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
train_list = []
val_list = []
val_acc = []
train_acc = []

for iter in range(1000):
    model.train()
    for idx, (param, target) in enumerate(train_dataloader):
        for k in param.keys():
            param[k] = param[k].cuda()
        target = target.cuda().reshape(-1)
        pred = model(param)
        loss = loss_func(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(pred, 1)
        acc = torch.sum(predicted==target) / target.shape[0]

        train_list.append(loss.item())
        train_acc.append(acc.item())

    model.eval()
    for idx, (param, target) in enumerate(val_dataloader):
        for k in param.keys():
            param[k] = param[k].cuda()
        target = target.cuda().reshape(-1)
        pred = model(param)
        _, predicted = torch.max(pred, 1)
        acc_v = torch.sum(predicted==target) / target.shape[0]
        val_loss = loss_func(pred, target)
        val_list.append(val_loss.item())
        val_acc.append(acc_v.item())
    
    print(f"[{iter+1}/1000] Train Acc : {acc:.4f}, Train Loss : {loss:.4f} Val Acc : {acc_v:.4f}, Val Loss : {val_loss:.4f}")

print("hi")