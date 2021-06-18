import torch.nn as nn
import torch

class FunctionWrapperV2(nn.Module):
    def __init__(self, run_func, run_args):
        super(FunctionWrapperV2, self).__init__()
        self.run_func = run_func
        self.run_args = run_args
    def forward(self, x):
        return self.run_func(x, *self.run_args)

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

class Flatten(nn.Module):
    def __init__(self, dim=-1):
        super(Flatten, self).__init__()
        self.dim = dim
    def forward(self, x):
        x = x.view(x.size(0), self.dim)
        return x

class Cat(nn.Module):
    def __init__(self, dim=1):
        super(Cat, self).__init__()
        self.dim = dim
    def forward(self, x):
        return torch.cat(x, dim=self.dim)

class Add2(nn.Module):
    def __init__(self):
        super(Add2, self).__init__()
    def forward(self, x):
        assert type(x) == list and len(x) == 2
        return x[0] + x[1]

class Mul2(nn.Module):
    def __init__(self):
        super(Mul2, self).__init__()
    def forward(self, x):
        assert type(x) == list and len(x) == 2
        return x[0] * x[1]

class ListConstruct(nn.Module):
    def __init__(self):
        super(ListConstruct, self).__init__()
    def forward(self, x):
        # assume x is a list
        return list(x)

class TupleConstruct(nn.Module):
    def __init__(self):
        super(TupleConstruct, self).__init__()
    def forward(self, x):
        # assume x is a list
        return tuple(x)

class TupleIndexing(nn.Module):
    def __init__(self, index):
        super(TupleIndexing, self).__init__()
        self.index = index
    def forward(self, x):
        # assume x is a tuple
        return x[self.index]

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
