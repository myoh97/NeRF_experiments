import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class Baseline(nn.Module):
    def __init__(self, len_w = 593408, len_b = 2436):
        super(Baseline, self).__init__()

        self.linear_weight = nn.Linear(len_w, 256)
        self.weight_bn = nn.BatchNorm1d(256, track_running_stats=False)
        
        self.linear_bias = nn.Linear(len_b, 8)
        self.bias_bn = nn.BatchNorm1d(8, track_running_stats=False)
        
        self.hidden_layer1 = nn.Linear(264, 256)
        self.hidden1_bn = nn.BatchNorm1d(256, track_running_stats=False)
        
        self.hidden_layer2 = nn.Linear(256, 256)
        self.hidden2_bn = nn.BatchNorm1d(256, track_running_stats=False)
        
        self.classifier = nn.Linear(256, 3) # 8 * 6(color)
        
    def forward(self, w, b):
        w = F.relu(self.weight_bn(self.linear_weight(w)))
        b = F.relu(self.bias_bn(self.linear_bias(b)))
        h = torch.cat([w,b],dim=1)
        h = F.relu(self.hidden1_bn(self.hidden_layer1(h)))
        h = F.relu(self.hidden2_bn(self.hidden_layer2(h)))
        output = self.classifier(h)

        return output

class Baseline_light(nn.Module):
    def __init__(self):
        super(Baseline_light, self).__init__()

        self.linear_weight = nn.Linear(593408, 128)
        self.weight_bn = nn.BatchNorm1d(128, track_running_stats=False)
        
        self.linear_bias = nn.Linear(2436, 8)
        self.bias_bn = nn.BatchNorm1d(8, track_running_stats=False)
        
        self.hidden_layer1 = nn.Linear(128+8, 128)
        self.hidden1_bn = nn.BatchNorm1d(128, track_running_stats=False)
        
        self.hidden_layer2 = nn.Linear(128, 128)
        self.hidden2_bn = nn.BatchNorm1d(128, track_running_stats=False)
        
        self.classifier = nn.Linear(128, 8) # 8 * 6(color)
        
    def forward(self, w, b):
        w = F.relu(self.weight_bn(self.linear_weight(w)))
        b = F.relu(self.bias_bn(self.linear_bias(b)))
        h = torch.cat([w,b],dim=1)
        h = F.relu(self.hidden1_bn(self.hidden_layer1(h)))
        h = F.relu(self.hidden2_bn(self.hidden_layer2(h)))
        output = self.classifier(h)

        return output

class Baseline_light_drop(nn.Module):
    def __init__(self):
        super(Baseline_light, self).__init__()

        self.linear_weight = nn.Linear(593408, 128)
        self.weight_bn = nn.BatchNorm1d(128, track_running_stats=False)
        
        self.linear_bias = nn.Linear(2436, 8)
        self.bias_bn = nn.BatchNorm1d(8, track_running_stats=False)
        
        self.hidden_layer1 = nn.Linear(128+8, 128)
        self.hidden1_bn = nn.BatchNorm1d(128, track_running_stats=False)
        
        self.hidden_layer2 = nn.Linear(128, 128)
        self.hidden2_bn = nn.BatchNorm1d(128, track_running_stats=False)
        
        self.classifier = nn.Linear(128, 8) # 8 * 6(color)
        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, w, b):
        w = F.relu(self.weight_bn(self.linear_weight(w)))
        b = F.relu(self.bias_bn(self.linear_bias(b)))
        h = torch.cat([w,b],dim=1)
        h = F.relu(self.hidden1_bn(self.hidden_layer1(h)))
        h = self.dropout(h)
        h = F.relu(self.hidden2_bn(self.hidden_layer2(h)))
        output = self.classifier(h)

        return output