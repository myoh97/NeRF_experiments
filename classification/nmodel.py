import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FCLayer_woBN(nn.Module):
    def __init__(self, batchsize):
        super(FCLayer_woBN, self).__init__()

        self.linear_weight = nn.Linear(593408, 256)

        self.linear_bias = nn.Linear(2436, 8)

        self.hidden_layer1 = nn.Linear(264, 256)
        self.hidden_layer2 = nn.Linear(256, 256)
        self.classifier = nn.Linear(256, 8) # 8 * 6(color)

        self.batchsize = batchsize
    
    def forward(self, x):
        weights = []
        biases = []
        weights_result = []
        biases_result = []
        for k in x.keys():
            if k[-4:] != "bias":
                weights.append(x[k].reshape(self.batchsize, -1))
            else:
                biases.append(x[k].reshape(self.batchsize, -1))

        w = torch.cat(weights,dim=1)
        b = torch.cat(biases, dim=1)

        h = torch.cat((F.relu(self.linear_weight(w)), F.relu(self.linear_bias(b))),dim=1)
        h = F.relu(self.hidden_layer1(h))
        h = F.relu(self.hidden_layer2(h))
        output = self.classifier(h)

        return output