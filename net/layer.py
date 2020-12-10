import torch.nn as nn
import torch

class BasicIdentity(nn.Module):
    def __init__(self):
        super(BasicIdentity, self).__init__()
    def forward(self, x):
        return x


class BasicAdd(nn.Module):
    def __init__(self):
        super(BasicAdd, self).__init__()
        self.require_input = 2
    def forward(self, x):
        # assume x is a list
        return x[0] + x[1]

class BasicAddRelu(nn.Module):
    def __init__(self):
        super(BasicAddRelu, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.require_input = 2
    def forward(self, x):
        # assume x is a list
        return self.relu(x[0] + x[1])

class BasicAddReluAvgPool(nn.Module):
    def __init__(self):
        super(BasicAddReluAvgPool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.require_input = 2
    def forward(self, x):
        # assume x is a list
        return self.avgpool(self.relu(x[0] + x[1]))


class BasicFCReshape(nn.Module):
    def __init__(self, fc):
        super(BasicFCReshape, self).__init__()
        self.fc = fc
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

class BasicReshape(nn.Module):
    def __init__(self):
        super(BasicReshape, self).__init__()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class BasicCatDim1(nn.Module):
    def __init__(self):
        super(BasicCatDim1, self).__init__()
    def forward(self, x):
        return torch.cat(x, 1)

# todo: cannot pickle function wrapper
class FunctionWrapper(nn.Module):
    def __init__(self, run_func):
        super(FunctionWrapper, self).__init__()
        self.run_func = run_func
    def forward(self, x):
        return self.run_func(x)
