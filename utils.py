import torch
import numpy as np
import random
import pickle

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