from torchvision.models.inception import InceptionAux
from torchvision.models.inception import InceptionA as torchInceptionA
from torchvision.models.inception import InceptionB as torchInceptionB
from torchvision.models.inception import InceptionC as torchInceptionC
from torchvision.models.inception import InceptionD as torchInceptionD
from torchvision.models.inception import InceptionE as torchInceptionE
from torchvision.models.inception import BasicConv2d as torchBasicConv2d
from torchvision.models.inception import load_state_dict_from_url, model_urls
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from functools import partial
from .layer import FunctionWrapper, BasicIdentity, BasicCatDim1, BasicReshape



def inception_v3(pretrained=False, progress=True, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    kwargs['aux_logits'] = False
    kwargs['transform_input'] = False
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' in kwargs:
            original_aux_logits = kwargs['aux_logits']
            kwargs['aux_logits'] = True
        else:
            original_aux_logits = True
        model = Inception3(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['inception_v3_google'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.AuxLogits
        return model

    return Inception3(**kwargs)


class Inception3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False,
                 inception_blocks=None, init_weights=True, drop_rate=0.5):
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.drop_rate = drop_rate
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.Mixed_5b = inception_a(192, pool_features=32, conv_block=BasicConv2d)
        self.Mixed_5c = inception_a(256, pool_features=64, conv_block=BasicConv2d)
        self.Mixed_5d = inception_a(288, pool_features=64, conv_block=BasicConv2d)
        self.Mixed_6a = inception_b(288, conv_block=BasicConv2d)
        self.Mixed_6b = inception_c(768, channels_7x7=128, conv_block=BasicConv2d)
        self.Mixed_6c = inception_c(768, channels_7x7=160, conv_block=BasicConv2d)
        self.Mixed_6d = inception_c(768, channels_7x7=160, conv_block=BasicConv2d)
        self.Mixed_6e = inception_c(768, channels_7x7=192, conv_block=BasicConv2d)
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768, conv_block=BasicConv2d)
        self.Mixed_7b = inception_e(1280, conv_block=BasicConv2d)
        self.Mixed_7c = inception_e(2048, conv_block=BasicConv2d)
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def parse_graph(self, x):
        G = nx.MultiDiGraph()
        source = 0
        vertex_id = 0
        G.add_node(vertex_id, cost=x.numel())

        if self.transform_input:
            op = FunctionWrapper(self._transform_input)
            x = op(x)
            G.add_node(vertex_id + 1, cost=x.numel())
            G.add_edge(x, vertex_id + 1, cost=0, module=op)
            vertex_id += 1

        G, x, vertex_id = self.Conv2d_1a_3x3.parse_graph(G, x, vertex_id, vertex_id)
        G, x, vertex_id = self.Conv2d_2a_3x3.parse_graph(G, x, vertex_id, vertex_id)
        G, x, vertex_id = self.Conv2d_2b_3x3.parse_graph(G, x, vertex_id, vertex_id)

        op = FunctionWrapper(partial(F.max_pool2d, kernel_size=3, stride=2))
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel())
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        G, x, vertex_id = self.Conv2d_3b_1x1.parse_graph(G, x, vertex_id, vertex_id)
        G, x, vertex_id = self.Conv2d_4a_3x3.parse_graph(G, x, vertex_id, vertex_id)

        op = FunctionWrapper(partial(F.max_pool2d, kernel_size=3, stride=2))
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel())
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        G, x, vertex_id = self.Mixed_5b.parse_graph(G, x, vertex_id, vertex_id)
        G, x, vertex_id = self.Mixed_5c.parse_graph(G, x, vertex_id, vertex_id)
        G, x, vertex_id = self.Mixed_5d.parse_graph(G, x, vertex_id, vertex_id)
        G, x, vertex_id = self.Mixed_6a.parse_graph(G, x, vertex_id, vertex_id)
        G, x, vertex_id = self.Mixed_6b.parse_graph(G, x, vertex_id, vertex_id)
        G, x, vertex_id = self.Mixed_6c.parse_graph(G, x, vertex_id, vertex_id)
        G, x, vertex_id = self.Mixed_6d.parse_graph(G, x, vertex_id, vertex_id)
        G, x, vertex_id = self.Mixed_6e.parse_graph(G, x, vertex_id, vertex_id)
        G, x, vertex_id = self.Mixed_7a.parse_graph(G, x, vertex_id, vertex_id)
        G, x, vertex_id = self.Mixed_7b.parse_graph(G, x, vertex_id, vertex_id)
        G, x, vertex_id = self.Mixed_7c.parse_graph(G, x, vertex_id, vertex_id)

        op = FunctionWrapper(partial(F.adaptive_avg_pool2d, output_size=(1, 1)))
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel())
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        op = FunctionWrapper(partial(F.dropout, training=self.training, p=self.drop_rate))
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel())
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        op = nn.Sequential(BasicReshape(), self.fc)
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel())
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        target = vertex_id

        return G, source, target

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux = self.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training, p=self.drop_rate)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

    def forward(self, x):
        x = self._transform_input(x)
        x, aux = self._forward(x)
        return x



class InceptionA(torchInceptionA):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__(in_channels, pool_features, conv_block=conv_block)

    def parse_graph(self, G, x, input_id, vertex_id):
        identity = BasicIdentity()
        concat = BasicCatDim1()

        G, branch1x1, vertex_id = self.branch1x1.parse_graph(G, x, input_id, vertex_id)
        branch1x1_vid = vertex_id

        G, branch5x5, vertex_id = self.branch5x5_1.parse_graph(G, x, input_id, vertex_id)
        branch5x5_vid = vertex_id
        G, branch5x5, vertex_id = self.branch5x5_2.parse_graph(G, branch5x5, branch5x5_vid, vertex_id)
        branch5x5_vid = vertex_id

        G, branch3x3dbl, vertex_id = self.branch3x3dbl_1.parse_graph(G, x, input_id, vertex_id)
        branch3x3dbl_vid = vertex_id
        G, branch3x3dbl, vertex_id = self.branch3x3dbl_2.parse_graph(G, branch3x3dbl, branch3x3dbl_vid, vertex_id)
        branch3x3dbl_vid = vertex_id
        G, branch3x3dbl, vertex_id = self.branch3x3dbl_3.parse_graph(G, branch3x3dbl, branch3x3dbl_vid, vertex_id)
        branch3x3dbl_vid = vertex_id

        op = FunctionWrapper(partial(F.avg_pool2d, kernel_size=3, stride=1, padding=1))
        branch_pool = op(x)
        G.add_node(vertex_id + 1, cost=branch_pool.numel())
        G.add_edge(input_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1
        branch_pool_vid = vertex_id

        G, branch_pool, vertex_id = self.branch_pool.parse_graph(G, branch_pool, branch_pool_vid, vertex_id)
        branch_pool_vid = vertex_id


        outputs = concat([branch1x1, branch5x5, branch3x3dbl, branch_pool])
        G.add_node(vertex_id + 1, cost=outputs.numel(), transition=concat,
                   transition_input_order=[(branch1x1_vid, 0), (branch5x5_vid, 0), (branch3x3dbl_vid, 0), (branch_pool_vid, 0)])
        G.add_edge(branch1x1_vid, vertex_id + 1, cost=0, module=identity)
        G.add_edge(branch5x5_vid, vertex_id + 1, cost=0, module=identity)
        G.add_edge(branch3x3dbl_vid, vertex_id + 1, cost=0, module=identity)
        G.add_edge(branch_pool_vid, vertex_id + 1, cost=0, module=identity)
        vertex_id += 1

        return G, outputs, vertex_id


class InceptionB(torchInceptionB):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__(in_channels, conv_block=conv_block)

    def parse_graph(self, G, x, input_id, vertex_id):
        identity = BasicIdentity()
        concat = BasicCatDim1()

        G, branch3x3, vertex_id = self.branch3x3.parse_graph(G, x, input_id, vertex_id)
        branch3x3_vid = vertex_id

        G, branch3x3dbl, vertex_id = self.branch3x3dbl_1.parse_graph(G, x, input_id, vertex_id)
        branch3x3dbl_vid = vertex_id
        G, branch3x3dbl, vertex_id = self.branch3x3dbl_2.parse_graph(G, branch3x3dbl, branch3x3dbl_vid, vertex_id)
        branch3x3dbl_vid = vertex_id
        G, branch3x3dbl, vertex_id = self.branch3x3dbl_3.parse_graph(G, branch3x3dbl, branch3x3dbl_vid, vertex_id)
        branch3x3dbl_vid = vertex_id

        op = FunctionWrapper(partial(F.max_pool2d, kernel_size=3, stride=2))
        branch_pool = op(x)
        G.add_node(vertex_id + 1, cost=branch_pool.numel())
        G.add_edge(input_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1
        branch_pool_vid = vertex_id


        outputs = concat([branch3x3, branch3x3dbl, branch_pool])
        G.add_node(vertex_id + 1, cost=outputs.numel(), transition=concat,
                   transition_input_order=[(branch3x3_vid, 0), (branch3x3dbl_vid, 0), (branch_pool_vid, 0)])
        G.add_edge(branch3x3_vid, vertex_id + 1, cost=0, module=identity)
        G.add_edge(branch3x3dbl_vid, vertex_id + 1, cost=0, module=identity)
        G.add_edge(branch_pool_vid, vertex_id + 1, cost=0, module=identity)
        vertex_id += 1

        return G, outputs, vertex_id


class InceptionC(torchInceptionC):

    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__(in_channels, channels_7x7, conv_block=conv_block)

    def parse_graph(self, G, x, input_id, vertex_id):
        identity = BasicIdentity()
        concat = BasicCatDim1()

        G, branch1x1, vertex_id = self.branch1x1.parse_graph(G, x, input_id, vertex_id)
        branch1x1_vid = vertex_id

        G, branch7x7, vertex_id = self.branch7x7_1.parse_graph(G, x, input_id, vertex_id)
        branch7x7_vid = vertex_id
        G, branch7x7, vertex_id = self.branch7x7_2.parse_graph(G, branch7x7, branch7x7_vid, vertex_id)
        branch7x7_vid = vertex_id
        G, branch7x7, vertex_id = self.branch7x7_3.parse_graph(G, branch7x7, branch7x7_vid, vertex_id)
        branch7x7_vid = vertex_id

        G, branch7x7dbl, vertex_id = self.branch7x7dbl_1.parse_graph(G, x, input_id, vertex_id)
        branch7x7dbl_vid = vertex_id
        G, branch7x7dbl, vertex_id = self.branch7x7dbl_2.parse_graph(G, branch7x7dbl, branch7x7dbl_vid, vertex_id)
        branch7x7dbl_vid = vertex_id
        G, branch7x7dbl, vertex_id = self.branch7x7dbl_3.parse_graph(G, branch7x7dbl, branch7x7dbl_vid, vertex_id)
        branch7x7dbl_vid = vertex_id
        G, branch7x7dbl, vertex_id = self.branch7x7dbl_4.parse_graph(G, branch7x7dbl, branch7x7dbl_vid, vertex_id)
        branch7x7dbl_vid = vertex_id
        G, branch7x7dbl, vertex_id = self.branch7x7dbl_5.parse_graph(G, branch7x7dbl, branch7x7dbl_vid, vertex_id)
        branch7x7dbl_vid = vertex_id

        op = FunctionWrapper(partial(F.avg_pool2d, kernel_size=3, stride=1, padding=1))
        branch_pool = op(x)
        G.add_node(vertex_id + 1, cost=branch_pool.numel())
        G.add_edge(input_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1
        branch_pool_vid = vertex_id

        G, branch_pool, vertex_id = self.branch_pool.parse_graph(G, branch_pool, branch_pool_vid, vertex_id)
        branch_pool_vid = vertex_id


        outputs = concat([branch1x1, branch7x7, branch7x7dbl, branch_pool])
        G.add_node(vertex_id + 1, cost=outputs.numel(), transition=concat,
                   transition_input_order=[(branch1x1_vid, 0), (branch7x7_vid, 0), (branch7x7dbl_vid, 0), (branch_pool_vid, 0)])
        G.add_edge(branch1x1_vid, vertex_id + 1, cost=0, module=identity)
        G.add_edge(branch7x7_vid, vertex_id + 1, cost=0, module=identity)
        G.add_edge(branch7x7dbl_vid, vertex_id + 1, cost=0, module=identity)
        G.add_edge(branch_pool_vid, vertex_id + 1, cost=0, module=identity)
        vertex_id += 1

        return G, outputs, vertex_id


class InceptionD(torchInceptionD):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__(in_channels, conv_block=conv_block)


    def parse_graph(self, G, x, input_id, vertex_id):
        identity = BasicIdentity()
        concat = BasicCatDim1()

        G, branch3x3, vertex_id = self.branch3x3_1.parse_graph(G, x, input_id, vertex_id)
        branch3x3_vid = vertex_id
        G, branch3x3, vertex_id = self.branch3x3_2.parse_graph(G, branch3x3, branch3x3_vid, vertex_id)
        branch3x3_vid = vertex_id

        G, branch7x7x3, vertex_id = self.branch7x7x3_1.parse_graph(G, x, input_id, vertex_id)
        branch7x7x3_vid = vertex_id
        G, branch7x7x3, vertex_id = self.branch7x7x3_2.parse_graph(G, branch7x7x3, branch7x7x3_vid, vertex_id)
        branch7x7x3_vid = vertex_id
        G, branch7x7x3, vertex_id = self.branch7x7x3_3.parse_graph(G, branch7x7x3, branch7x7x3_vid, vertex_id)
        branch7x7x3_vid = vertex_id
        G, branch7x7x3, vertex_id = self.branch7x7x3_4.parse_graph(G, branch7x7x3, branch7x7x3_vid, vertex_id)
        branch7x7x3_vid = vertex_id

        op = FunctionWrapper(partial(F.max_pool2d, kernel_size=3, stride=2))
        branch_pool = op(x)
        G.add_node(vertex_id + 1, cost=branch_pool.numel())
        G.add_edge(input_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1
        branch_pool_vid = vertex_id


        outputs = concat([branch3x3, branch7x7x3, branch_pool])
        G.add_node(vertex_id + 1, cost=outputs.numel(), transition=concat,
                   transition_input_order=[(branch3x3_vid, 0), (branch7x7x3_vid, 0), (branch_pool_vid, 0)])
        G.add_edge(branch3x3_vid, vertex_id + 1, cost=0, module=identity)
        G.add_edge(branch7x7x3_vid, vertex_id + 1, cost=0, module=identity)
        G.add_edge(branch_pool_vid, vertex_id + 1, cost=0, module=identity)
        vertex_id += 1

        return G, outputs, vertex_id


class InceptionE(torchInceptionE):


    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__(in_channels, conv_block=conv_block)

    def parse_graph(self, G, x, input_id, vertex_id):
        identity = BasicIdentity()
        concat = BasicCatDim1()

        G, branch1x1, vertex_id = self.branch1x1.parse_graph(G, x, input_id, vertex_id)
        branch1x1_vid = vertex_id
        G, branch3x3, vertex_id = self.branch3x3_1.parse_graph(G, x, input_id, vertex_id)
        branch3x3_vid = vertex_id

        G, branch3x3_2a, vertex_id = self.branch3x3_2a.parse_graph(G, branch3x3, branch3x3_vid, vertex_id)
        branch3x3_2a_vid = vertex_id
        G, branch3x3_2b, vertex_id = self.branch3x3_2b.parse_graph(G, branch3x3, branch3x3_vid, vertex_id)
        branch3x3_2b_vid = vertex_id

        branch3x3 = concat([branch3x3_2a, branch3x3_2b])
        G.add_node(vertex_id + 1, cost=branch3x3.numel(), transition=concat, transition_input_order=[(branch3x3_2a_vid, 0), (branch3x3_2b_vid, 0)])
        G.add_edge(branch3x3_2a_vid, vertex_id + 1, cost=0, module=identity)
        G.add_edge(branch3x3_2b_vid, vertex_id + 1, cost=0, module=identity)
        vertex_id += 1
        branch3x3_vid = vertex_id

        G, branch3x3dbl, vertex_id = self.branch3x3dbl_1.parse_graph(G, x, input_id, vertex_id)
        branch3x3dbl_vid = vertex_id
        G, branch3x3dbl, vertex_id = self.branch3x3dbl_2.parse_graph(G, branch3x3dbl, branch3x3dbl_vid, vertex_id)
        branch3x3dbl_vid = vertex_id

        G, branch3x3dbl_3a, vertex_id = self.branch3x3dbl_3a.parse_graph(G, branch3x3dbl, branch3x3dbl_vid, vertex_id)
        branch3x3dbl_3a_vid = vertex_id
        G, branch3x3dbl_3b, vertex_id = self.branch3x3dbl_3b.parse_graph(G, branch3x3dbl, branch3x3dbl_vid, vertex_id)
        branch3x3dbl_3b_vid = vertex_id

        branch3x3dbl = concat([branch3x3dbl_3a, branch3x3dbl_3b])
        G.add_node(vertex_id + 1, cost=branch3x3dbl.numel(), transition=concat,
                   transition_input_order=[(branch3x3dbl_3a_vid, 0), (branch3x3dbl_3b_vid, 0)])
        G.add_edge(branch3x3dbl_3a_vid, vertex_id + 1, cost=0, module=identity)
        G.add_edge(branch3x3dbl_3b_vid, vertex_id + 1, cost=0, module=identity)
        vertex_id += 1
        branch3x3dbl_vid = vertex_id

        op = FunctionWrapper(partial(F.avg_pool2d, kernel_size=3, stride=1, padding=1))
        branch_pool = op(x)
        G.add_node(vertex_id + 1, cost=branch_pool.numel())
        G.add_edge(input_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1
        branch_pool_vid = vertex_id

        G, branch_pool, vertex_id = self.branch_pool.parse_graph(G, branch_pool, branch_pool_vid, vertex_id)
        branch_pool_vid = vertex_id

        outputs = concat([branch1x1, branch3x3, branch3x3dbl, branch_pool])
        G.add_node(vertex_id + 1, cost=outputs.numel(), transition=concat,
                   transition_input_order=[(branch1x1_vid, 0), (branch3x3_vid, 0), (branch3x3dbl_vid, 0), (branch_pool_vid, 0)])
        G.add_edge(branch1x1_vid, vertex_id + 1, cost=0, module=identity)
        G.add_edge(branch3x3_vid, vertex_id + 1, cost=0, module=identity)
        G.add_edge(branch3x3dbl_vid, vertex_id + 1, cost=0, module=identity)
        G.add_edge(branch_pool_vid, vertex_id + 1, cost=0, module=identity)
        vertex_id += 1

        return G, outputs, vertex_id




class BasicConv2d(torchBasicConv2d):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__(in_channels, out_channels, **kwargs)

    def parse_graph(self, G, x, input_id, vertex_id):
        if G is None:
            G = nx.MultiDiGraph()
            vertex_id = 0
            input_id = 0
            G.add_node(vertex_id, cost=x.numel())

        relu = FunctionWrapper(partial(F.relu, inplace=True))

        op = self.conv
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel())
        G.add_edge(input_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        op = nn.Sequential(self.bn, relu)
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel())
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1


        return G, x, vertex_id
