from torchvision.models.vgg import VGG as torchVGG
from torchvision.models.vgg import make_layers, model_urls
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import networkx as nx
from .layer import BasicFCReshape

class VGG(torchVGG):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__(features, num_classes=num_classes)

    def parse_graph(self, x):
        G, source, vertex_count, x = self.parse_layer_graph(x)

        fcReshape = BasicFCReshape(self.classifier[0])
        op = nn.Sequential(fcReshape, self.classifier[1])
        x = op(x)
        G.add_node(vertex_count + 1, cost=x.numel())
        G.add_edge(vertex_count, vertex_count + 1, cost=0, module=op)
        vertex_count += 1

        op = nn.Sequential(self.classifier[3], self.classifier[4])
        x = op(x)
        G.add_node(vertex_count + 1, cost=x.numel())
        G.add_edge(vertex_count, vertex_count + 1, cost=0, module=op)
        vertex_count += 1

        op = self.classifier[6]
        x = op(x)
        G.add_node(vertex_count + 1, cost=x.numel())
        G.add_edge(vertex_count, vertex_count + 1, cost=0, module=op)
        vertex_count += 1
        target = vertex_count

        return G, source, target

    def parse_layer_graph(self, x):
        cfg = self.cfg
        batch_norm = getattr(self, 'batch_norm', False)
        G = nx.MultiDiGraph()
        source = 0
        G.add_node(0, cost=x.numel())


        vertex_count = 0
        layer_counter = 0
        in_channels = 3
        for v in cfg:
            if v == 'M':
                op = self.features[layer_counter]
                layer_counter += 1
                x = op(x)
                G.add_node(vertex_count + 1, cost=x.numel())
                G.add_edge(vertex_count, vertex_count + 1, cost=0, module=op)
                vertex_count += 1
            else:
                if batch_norm:
                    op = nn.Sequential(self.features[layer_counter], self.features[layer_counter + 1],
                                       self.features[layer_counter + 2])
                    layer_counter += 3
                else:
                    op = nn.Sequential(self.features[layer_counter], self.features[layer_counter + 1])
                    layer_counter += 2
                x = op(x)

                G.add_node(vertex_count + 1, cost=x.numel())
                G.add_edge(vertex_count, vertex_count + 1, cost=0, module=op)

                vertex_count += 1
        target = vertex_count
        return G, source, target, x


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    model.cfg = cfg['A']

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    model.cfg = cfg['A']
    model.batch_norm = True
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    model.cfg = cfg['B']
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    model.cfg = cfg['B']
    model.batch_norm = True
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    model.cfg = cfg['D']
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    model.cfg = cfg['D']
    model.batch_norm = True
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    model.cfg = cfg['E']
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    model.cfg = cfg['E']
    model.batch_norm = True
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model