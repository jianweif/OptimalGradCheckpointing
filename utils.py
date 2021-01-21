import torch
import numpy as np
import random
import pickle

def disable_dropout(arch, net):
    # manually turn off dropout to pass backward check
    if arch in ['alexnet']:
        net.classifier[0].p = 0
        net.classifier[3].p = 0
    elif arch in ['vgg11', 'vgg13', 'vgg16', 'vgg19']:
        net.classifier[2].p = 0
        net.classifier[5].p = 0
    elif arch in ['inception_v3']:
        net.drop_rate = 0
    elif arch in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
        for i, num_layers in enumerate(net.block_config):
            block = net.features.__getattr__('denseblock%d' % (i + 1))
            for i in range(num_layers):
                layer = block.__getattr__('denselayer%d' % (i + 1))
                layer.drop_rate = 0
    else:
        pass

def set_reproductibility(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def add_vertex_cost_to_edge(G):
    for edge_key in G.edges:
        _, target_key, _ = edge_key
        target_cost = G.nodes[target_key]['cost']
        edge_cost = G.edges[edge_key]['cost']
        G.edges[edge_key]['weight'] = target_cost + edge_cost

    return G

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)