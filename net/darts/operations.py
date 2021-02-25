import torch
import torch.nn as nn
from net.layer import BasicCatDim1, BasicIdentity

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: Conv7x1_1x7(C, stride, affine),
}

class Conv7x1_1x7(nn.Sequential):
    def __init__(self,  C, stride, affine):
        super(Conv7x1_1x7, self).__init__()
        self.add_module('relu', nn.ReLU(inplace=False))
        self.add_module('conv1', nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False))
        self.add_module('conv2', nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False))
        self.add_module('bn', nn.BatchNorm2d(C, affine=affine))

    def parse_graph(self, G, x, input_id, vertex_id):
        op = self.relu
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel(), shape=list(x.shape))
        G.add_edge(input_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        op = self.conv1
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel(), shape=list(x.shape))
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        op = self.conv2
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel(), shape=list(x.shape))
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        op = self.bn
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel(), shape=list(x.shape))
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        return G, x, vertex_id


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

    def parse_graph(self, G, x, input_id, vertex_id):
        # inplace=False for relu so not combining relu and conv2d into 1 op

        for i in range(len(self.op)):
            if i == 0:
                prev_id = input_id
            else:
                prev_id = vertex_id
            op = self.op[i]
            x = op(x)
            G.add_node(vertex_id + 1, cost=x.numel(), shape=list(x.shape))
            G.add_edge(prev_id, vertex_id + 1, cost=0, module=op)
            vertex_id += 1

        return G, x, vertex_id


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

    def parse_graph(self, G, x, input_id, vertex_id):
        # inplace=False for relu so not combining relu and conv2d into 1 op
        for i in range(len(self.op)):
            if i == 0:
                prev_id = input_id
            else:
                prev_id = vertex_id
            op = self.op[i]
            x = op(x)
            G.add_node(vertex_id + 1, cost=x.numel(), shape=list(x.shape))
            G.add_edge(prev_id, vertex_id + 1, cost=0, module=op)
            vertex_id += 1

        return G, x, vertex_id


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

    def parse_graph(self, G, x, input_id, vertex_id):
        # inplace=False for relu so not combining relu and conv2d into 1 op
        for i in range(len(self.op)):
            if i == 0:
                prev_id = input_id
            else:
                prev_id = vertex_id
            op = self.op[i]
            x = op(x)
            G.add_node(vertex_id + 1, cost=x.numel(), shape=list(x.shape))
            G.add_edge(prev_id, vertex_id + 1, cost=0, module=op)
            vertex_id += 1

        return G, x, vertex_id


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class Conv2Wrapper(nn.Module):
    '''
    a wrapper for the operaion conv_2(x[:, :, 1:, 1:]) in FactorizedReduce
    '''
    def __init__(self, conv):
        super(Conv2Wrapper, self).__init__()
        self.conv = conv

    def forward(self, x):
        return self.conv(x[:, :, 1:, 1:])


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)



    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

    def parse_graph(self, G, x, input_id, vertex_id):
        # inplace=False for relu so not combining relu and conv2d into 1 op

        op = self.relu
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel(), shape=list(x.shape))
        G.add_edge(input_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        input_id = vertex_id

        op1 = self.conv_1
        x1 = op1(x)
        G.add_node(vertex_id + 1, cost=x1.numel(), shape=list(x1.shape))
        G.add_edge(input_id, vertex_id + 1, cost=0, module=op1)
        vertex_id += 1
        x1_id = vertex_id



        op2 = Conv2Wrapper(conv=self.conv_2)
        x2 = op2(x)
        G.add_node(vertex_id + 1, cost=x2.numel(), shape=list(x2.shape))
        G.add_edge(input_id, vertex_id + 1, cost=0, module=op2)
        vertex_id += 1
        x2_id = vertex_id

        op = BasicCatDim1()
        identity = BasicIdentity()
        x = op([x1, x2])
        G.add_node(vertex_id + 1, cost=x.numel(), transition=op, shape=list(x.shape))
        G.nodes[vertex_id + 1]['transition_input_order'] = []
        for id in [x1_id, x2_id]:
            edge_id = G.add_edge(id, vertex_id + 1, cost=0, module=identity)
            G.nodes[vertex_id + 1]['transition_input_order'].append((id, edge_id))
        vertex_id += 1

        op = self.bn
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel(), shape=list(x.shape))
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        return G, x, vertex_id

