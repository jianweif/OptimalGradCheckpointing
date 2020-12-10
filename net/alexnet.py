from torchvision.models.alexnet import AlexNet as torchAlexNet
from torchvision.models.alexnet import load_state_dict_from_url, model_urls
import torch.nn as nn
import networkx as nx
from .layer import BasicReshape

class AlexNet(torchAlexNet):

    def __init__(self, **kwargs):
        super(AlexNet, self).__init__(**kwargs)

    def parse_graph(self, x):
        G = nx.MultiDiGraph()
        source = 0
        G.add_node(0, cost=x.numel())

        vertex_count = 0

        op = nn.Sequential(*self.features[0:2])
        x = op(x)
        G.add_node(vertex_count + 1, cost=x.numel())
        G.add_edge(vertex_count, vertex_count + 1, cost=0, module=op)
        vertex_count += 1

        op = self.features[2]
        x = op(x)
        G.add_node(vertex_count + 1, cost=x.numel())
        G.add_edge(vertex_count, vertex_count + 1, cost=0, module=op)
        vertex_count += 1

        op = nn.Sequential(*self.features[3:5])
        x = op(x)
        G.add_node(vertex_count + 1, cost=x.numel())
        G.add_edge(vertex_count, vertex_count + 1, cost=0, module=op)
        vertex_count += 1

        op = self.features[5]
        x = op(x)
        G.add_node(vertex_count + 1, cost=x.numel())
        G.add_edge(vertex_count, vertex_count + 1, cost=0, module=op)
        vertex_count += 1

        op = nn.Sequential(*self.features[6:8])
        x = op(x)
        G.add_node(vertex_count + 1, cost=x.numel())
        G.add_edge(vertex_count, vertex_count + 1, cost=0, module=op)
        vertex_count += 1

        op = nn.Sequential(*self.features[8:10])
        x = op(x)
        G.add_node(vertex_count + 1, cost=x.numel())
        G.add_edge(vertex_count, vertex_count + 1, cost=0, module=op)
        vertex_count += 1

        op = nn.Sequential(*self.features[10:12])
        x = op(x)
        G.add_node(vertex_count + 1, cost=x.numel())
        G.add_edge(vertex_count, vertex_count + 1, cost=0, module=op)
        vertex_count += 1

        op = self.features[12]
        x = op(x)
        G.add_node(vertex_count + 1, cost=x.numel())
        G.add_edge(vertex_count, vertex_count + 1, cost=0, module=op)
        vertex_count += 1

        reshape_op = BasicReshape()
        op = nn.Sequential(self.avgpool, reshape_op)
        x = op(x)
        G.add_node(vertex_count + 1, cost=x.numel())
        G.add_edge(vertex_count, vertex_count + 1, cost=0, module=op)
        vertex_count += 1

        op = nn.Sequential(*self.classifier[0:3])
        x = op(x)
        G.add_node(vertex_count + 1, cost=x.numel())
        G.add_edge(vertex_count, vertex_count + 1, cost=0, module=op)
        vertex_count += 1

        op = nn.Sequential(*self.classifier[3:6])
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




def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model