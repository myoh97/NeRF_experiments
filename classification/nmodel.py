import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FCLayer(nn.Module):
    def __init__(self, batchsize):
        super(FCLayer, self).__init__()

        self.linear_weight = nn.Linear(593408, 256)

        self.linear_bias = nn.Linear(2436, 8)

        self.hidden_layer1 = nn.Linear(264, 256)
        self.hidden_layer2 = nn.Linear(256, 256)
        self.classifier = nn.Linear(256, 8) # 8 * 6(color)

        self.batchsize = batchsize
    
    def forward(self, w, b):
        # weights = []
        # biases = []
        # weights_result = []
        # biases_result = []
        # for k in x.keys():
        #     if k[-4:] != "bias":
        #         weights.append(x[k].reshape(self.batchsize, -1))
        #     else:
        #         biases.append(x[k].reshape(self.batchsize, -1))

        # w = torch.cat(weights,dim=1)
        # b = torch.cat(biases, dim=1)

        h = torch.cat((F.relu(self.linear_weight(w)), F.relu(self.linear_bias(b))),dim=1)
        h = F.relu(self.hidden_layer1(h))
        h = F.relu(self.hidden_layer2(h))
        output = self.classifier(h)

        return output

class FCLayer_light(nn.Module):
    def __init__(self, batchsize):
        super(FCLayer_light, self).__init__()

        self.linear_weight = nn.Linear(593408, 256)

        self.linear_bias = nn.Linear(2436, 8)

        self.hidden_layer1 = nn.Linear(264, 128)
        self.hidden_layer2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, 8) # 8 * 6(color)

        self.batchsize = batchsize
    
    def forward(self, w, b):
        # weights = []
        # biases = []
        # weights_result = []
        # biases_result = []
        # for k in x.keys():
        #     if k[-4:] != "bias":
        #         weights.append(x[k].reshape(self.batchsize, -1))
        #     else:
        #         biases.append(x[k].reshape(self.batchsize, -1))

        # w = torch.cat(weights,dim=1)
        # b = torch.cat(biases, dim=1)

        h = torch.cat((F.relu(self.linear_weight(w)), F.relu(self.linear_bias(b))),dim=1)
        h = F.relu(self.hidden_layer1(h))
        h = F.relu(self.hidden_layer2(h))
        output = self.classifier(h)

        return output

class FCLayer_heavy(nn.Module):
    def __init__(self, batchsize):
        super(FCLayer_heavy, self).__init__()

        self.linear_weight = nn.Linear(593408, 1024)

        self.linear_bias = nn.Linear(2436, 1024)

        self.hidden_layer1 = nn.Linear(2048, 1024)
        self.hidden_layer2 = nn.Linear(1024, 512)
        self.hidden_layer3 = nn.Linear(512, 256)
        self.hidden_layer4 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, 8) # 8 * 6(color)

        self.batchsize = batchsize
    
    def forward(self, w, b):
        h = torch.cat((F.relu(self.linear_weight(w)), F.relu(self.linear_bias(b))),dim=1)
        h = F.relu(self.hidden_layer1(h))
        h = F.relu(self.hidden_layer2(h))
        h = F.relu(self.hidden_layer3(h))
        h = F.relu(self.hidden_layer4(h))
        output = self.classifier(h)

        return output
       
class FCLayer_dropout(nn.Module):
    def __init__(self, batchsize):
        super(FCLayer_dropout, self).__init__()

        self.linear_weight = nn.Linear(593408, 256)
        self.linear_bias = nn.Linear(2436, 8)
        
        self.hidden_layer1 = nn.Linear(264, 256)
        self.hidden_layer2 = nn.Linear(256, 256)
        
        self.classifier = nn.Linear(256, 8) # 8 * 6(color)
        self.dropout = nn.Dropout(p=0.5)
        self.batchsize = batchsize
        
    def forward(self, w, b):
        w = F.relu(self.linear_weight(w))
        w = self.dropout(w)
        b = F.relu(self.linear_bias(b))
        b = self.dropout(b)
        h = torch.cat([w,b],dim=1)
        h = F.relu(self.hidden_layer1(h))
        h = self.dropout(h)
        h = F.relu(self.hidden_layer2(h))
        h = self.dropout(h)
        output = self.classifier(h)

        return output
    
class FCLayer_BN(nn.Module):
    def __init__(self, batchsize):
        super(FCLayer_BN, self).__init__()

        self.linear_weight = nn.Linear(593408, 256)
        self.weight_bn = nn.BatchNorm1d(256, affine=False)
        
        self.linear_bias = nn.Linear(2436, 8)
        self.bias_bn = nn.BatchNorm1d(8)
        
        self.hidden_layer1 = nn.Linear(264, 256)
        self.hidden1_bn = nn.BatchNorm1d(256)
        
        self.hidden_layer2 = nn.Linear(256, 256)
        self.hidden2_bn = nn.BatchNorm1d(256)
        
        self.classifier = nn.Linear(256, 8) # 8 * 6(color)
        self.batchsize = batchsize
        
    def forward(self, w, b):
        w = F.relu(self.weight_bn(self.linear_weight(w)))
        b = F.relu(self.bias_bn(self.linear_bias(b)))
        h = torch.cat([w,b],dim=1)
        h = F.relu(self.hidden1_bn(self.hidden_layer1(h)))
        h = F.relu(self.hidden2_bn(self.hidden_layer2(h)))
        output = self.classifier(h)

        return output
    
class FCLayer_BN_dropout(nn.Module):
    def __init__(self, batchsize):
        super(FCLayer_BN_dropout, self).__init__()

        self.linear_weight = nn.Linear(593408, 256)
        self.weight_bn = nn.BatchNorm1d(256)
        
        self.linear_bias = nn.Linear(2436, 8)
        self.bias_bn = nn.BatchNorm1d(8)
        
        self.hidden_layer1 = nn.Linear(264, 256)
        self.hidden1_bn = nn.BatchNorm1d(256)
        
        self.hidden_layer2 = nn.Linear(256, 256)
        self.hidden2_bn = nn.BatchNorm1d(256)
        
        self.classifier = nn.Linear(256, 8) # 8 * 6(color)
        self.dropout = nn.Dropout(p=0.5)
        self.batchsize = batchsize
        
    def forward(self, w, b):
        w = F.relu(self.weight_bn(self.linear_weight(w)))
        w = self.dropout(w)
        b = F.relu(self.bias_bn(self.linear_bias(b)))
        b = self.dropout(b)
        h = torch.cat([w,b],dim=1)
        h = F.relu(self.hidden1_bn(self.hidden_layer1(h)))
        h = self.dropout(h)
        h = F.relu(self.hidden2_bn(self.hidden_layer2(h)))
        h = self.dropout(h)
        output = self.classifier(h)

        return output
    
class FCLayer_IN(nn.Module):
    def __init__(self, batchsize):
        super(FCLayer_IN, self).__init__()

        self.linear_weight = nn.Linear(593408, 256)
        self.weight_in = nn.InstanceNorm1d(256)
        
        self.linear_bias = nn.Linear(2436, 8)
        self.bias_in = nn.InstanceNorm1d(8)
        
        self.hidden_layer1 = nn.Linear(264, 256)
        self.hidden1_in = nn.InstanceNorm1d(256)
        
        self.hidden_layer2 = nn.Linear(256, 256)
        self.hidden2_in = nn.InstanceNorm1d(256)
        
        self.classifier = nn.Linear(256, 8) # 8 * 6(color)
        self.batchsize = batchsize
        
    def forward(self, w, b):
        w = F.relu(self.weight_in(self.linear_weight(w)))
        b = F.relu(self.bias_in(self.linear_bias(b)))
        h = torch.cat([w,b],dim=1)
        h = F.relu(self.hidden1_in(self.hidden_layer1(h)))
        h = F.relu(self.hidden2_in(self.hidden_layer2(h)))
        output = self.classifier(h)

        return output
    
class FCLayer_IN_dropout(nn.Module):
    def __init__(self, batchsize):
        super(FCLayer_IN_dropout, self).__init__()

        self.linear_weight = nn.Linear(593408, 256)
        self.weight_in = nn.InstanceNorm1d(256)
        
        self.linear_bias = nn.Linear(2436, 8)
        self.bias_in = nn.InstanceNorm1d(8)
        
        self.hidden_layer1 = nn.Linear(264, 256)
        self.hidden1_in = nn.InstanceNorm1d(256)
        
        self.hidden_layer2 = nn.Linear(256, 256)
        self.hidden2_in = nn.InstanceNorm1d(256)
        
        self.classifier = nn.Linear(256, 8) # 8 * 6(color)
        self.dropout = nn.Dropout(p=0.5)
        
        self.batchsize = batchsize
        
    def forward(self, w, b):
        w = F.relu(self.weight_in(self.linear_weight(w)))        
        w = self.dropout(w)
        b = F.relu(self.bias_in(self.linear_bias(b)))
        b = self.dropout(b)
        h = torch.cat([w,b],dim=1)
        h = F.relu(self.hidden1_in(self.hidden_layer1(h)))
        h = self.dropout(h)
        h = F.relu(self.hidden2_in(self.hidden_layer2(h)))
        h = self.dropout(h)
        output = self.classifier(h)

        return output
    
class Test(nn.Module):
    def __init__(self, batchsize):
        super(Test, self).__init__()

        self.linear_weight = nn.Linear(593408, 256)

        self.linear_bias = nn.Linear(2436, 8)

        self.hidden_layer1 = nn.Linear(264, 256)
        self.hidden_layer2 = nn.Linear(256, 2048)
        self.layer = nn.Sequential(
            nn.Linear(2048, 2048),nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048)
        )
        self.classifier = nn.Linear(2048, 8) # 8 * 6(color)

        self.batchsize = batchsize
    
    def forward(self, w, b):
        h = torch.cat((F.relu(self.linear_weight(w)), F.relu(self.linear_bias(b))),dim=1)
        h = F.relu(self.hidden_layer1(h))
        h = F.relu(self.hidden_layer2(h))
        h = F.relu(self.layer(h))
        output = self.classifier(h)

        return output