import torch.nn as nn
import torch
import networkx as nx
from torchvision.models.resnet import BasicBlock as torchBasicBlock
from torchvision.models.resnet import Bottleneck as torchBottleneck
from torchvision.models.resnet import ResNet as torchResNet
from torchvision.models.resnet import model_urls, load_state_dict_from_url
from .layer import BasicAddRelu, BasicIdentity, BasicFCReshape


class BasicBlock(torchBasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__(inplanes, planes, stride, downsample, groups,
                 base_width, dilation, norm_layer)

    def parse_graph(self, G, x, vertex_id):
        basicAdd = BasicAddRelu()
        basicIdentity = BasicIdentity()

        identity = x

        op = self.conv1
        out = op(x)
        G.add_node(vertex_id + 1, cost=out.numel())
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)

        op = nn.Sequential(self.bn1, self.relu)
        out = op(out)
        G.add_node(vertex_id + 2, cost=out.numel())
        G.add_edge(vertex_id + 1, vertex_id + 2, cost=0, module=op)

        op = self.conv2
        out = op(out)
        G.add_node(vertex_id + 3, cost=out.numel())
        G.add_edge(vertex_id + 2, vertex_id + 3, cost=0, module=op)

        op = self.bn2
        out = op(out)
        G.add_node(vertex_id + 5, cost=out.numel())
        G.add_edge(vertex_id + 3, vertex_id + 5, cost=0, module=op)


        if self.downsample is not None:
            op = self.downsample[0]
            identity = op(x)
            G.add_node(vertex_id + 4, cost=identity.numel())
            G.add_edge(vertex_id, vertex_id + 4, cost=0, module=op)

            op = self.downsample[1]
            identity = op(identity)
            # G.add_node(vertex_id + 5, cost=identity.numel())
            G.add_edge(vertex_id + 4, vertex_id + 5, cost=0, module=op)
        else:
            G.add_edge(vertex_id, vertex_id + 5, cost=0, module=basicIdentity)

        out += identity
        out = self.relu(out)


        if self.downsample is not None:
            transition_op = basicAdd
            transition_input_order = [(vertex_id + 3, 0), (vertex_id + 4, 0)]
            G.nodes[vertex_id + 5]['transition'] = transition_op
            G.nodes[vertex_id + 5]['transition_input_order'] = transition_input_order
        else:
            transition_op = basicAdd
            transition_input_order = [(vertex_id + 3, 0), (vertex_id + 0, 0)]
            G.nodes[vertex_id + 5]['transition'] = transition_op
            G.nodes[vertex_id + 5]['transition_input_order'] = transition_input_order

        return G, out, vertex_id + 5


class Bottleneck(torchBottleneck):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__(inplanes, planes, stride, downsample, groups,
                 base_width, dilation, norm_layer)

    def parse_graph(self, G, x, vertex_id):
        basicAdd = BasicAddRelu()
        basicIdentity = BasicIdentity()

        identity = x

        op = self.conv1
        out = op(x)
        G.add_node(vertex_id + 1, cost=out.numel())
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)

        op = nn.Sequential(self.bn1, self.relu)
        out = op(out)
        G.add_node(vertex_id + 2, cost=out.numel())
        G.add_edge(vertex_id + 1, vertex_id + 2, cost=0, module=op)

        op = self.conv2
        out = op(out)
        G.add_node(vertex_id + 3, cost=out.numel())
        G.add_edge(vertex_id + 2, vertex_id + 3, cost=0, module=op)

        op = nn.Sequential(self.bn2, self.relu)
        out = op(out)
        G.add_node(vertex_id + 4, cost=out.numel())
        G.add_edge(vertex_id + 3, vertex_id + 4, cost=0, module=op)

        op = self.conv3
        out = op(out)
        G.add_node(vertex_id + 5, cost=out.numel())
        G.add_edge(vertex_id + 4, vertex_id + 5, cost=0, module=op)

        op = self.bn3
        out = op(out)
        G.add_node(vertex_id + 7, cost=out.numel())
        G.add_edge(vertex_id + 5, vertex_id + 7, cost=0, module=op)


        if self.downsample is not None:
            op = self.downsample[0]
            identity = op(x)
            G.add_node(vertex_id + 6, cost=identity.numel())
            G.add_edge(vertex_id, vertex_id + 6, cost=0, module=op)

            op = self.downsample[1]
            identity = op(identity)
            # G.add_node(vertex_id + 7, cost=identity.numel())
            G.add_edge(vertex_id + 6, vertex_id + 7, cost=0, module=op)
        else:
            G.add_edge(vertex_id, vertex_id + 7, cost=0, module=basicIdentity)

        out += identity
        out = self.relu(out)


        if self.downsample is not None:
            transition_op = basicAdd
            transition_input_order = [(vertex_id + 5, 0), (vertex_id + 6, 0)]
            G.nodes[vertex_id + 7]['transition'] = transition_op
            G.nodes[vertex_id + 7]['transition_input_order'] = transition_input_order
        else:
            transition_op = basicAdd
            transition_input_order = [(vertex_id + 5, 0), (vertex_id + 0, 0)]
            G.nodes[vertex_id + 7]['transition'] = transition_op
            G.nodes[vertex_id + 7]['transition_input_order'] = transition_input_order

        return G, out, vertex_id + 7


class ResNet(torchResNet):

    def __init__(self, block, layers, **kwargs):
        super(ResNet, self).__init__(block, layers, **kwargs)

    def parse_graph(self, x):
        G = nx.MultiDiGraph()
        source = 0
        vertex_id = 0
        G.add_node(vertex_id, cost=x.numel())

        op = self.conv1
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel())
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        op = nn.Sequential(self.bn1, self.relu)
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel())
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        op = self.maxpool
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel())
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                G, x, vertex_id = block.parse_graph(G, x, vertex_id)

        op = self.avgpool
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel())
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        op = BasicFCReshape(self.fc)
        x = op(x)
        G.add_node(vertex_id + 1, cost=x.numel())
        G.add_edge(vertex_id, vertex_id + 1, cost=0, module=op)
        vertex_id += 1

        target = vertex_id

        return G, source, target


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
