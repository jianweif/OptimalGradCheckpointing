from torch import nn as nn
from torch.utils.checkpoint import checkpoint
from queue import Queue
import networkx as nx
import torch
from functools import partial

def tuple_to_dict(t):
    l = list(t)
    num = len(l) // 3
    d = {}
    for i in range(num):
        tensor, s, ind = t[i * 3], t[i * 3 + 1], t[i * 3 + 2]
        d[(int(s), int(ind))] = (tensor, s, ind)
    return d

def set_graph_training(graph, train=True):
    for e in graph.edges:
        module = graph.edges[e]['module']
        if isinstance(module, Segment):
            set_graph_training(module.G, train=train)
        else:
            if train:
                graph.edges[e]['module'].train()
            else:
                graph.edges[e]['module'].eval()

def replace_subgraph(graph1, graph2, source, target, id):
    '''
    replace subgraph in graph1 with graph2
    :param graph1: networkx DiGraph
    :param graph2: networkx DiGraph
    :param source: source vertex in graph1
    :param target: target vertex in graph1
    :param id: if None, meaning source and target is not connected, else specify the connection id
    :return:
    '''
    if source not in graph1.nodes or target not in graph1.nodes:
        raise ValueError
    if id is None:
        nodes1 = set(nx.ancestors(graph1, target))
        nodes2 = set(nx.descendants(graph1, source))
        nodes = (nodes1.intersection(nodes2)).union(set({source, target}))
        edges_add_back = {}
        for node in nodes:
            for p in graph1.predecessors(node):
                if p not in nodes:
                    es = graph1.get_edge_data(p, node)
                    if es is not None:
                        for e in es:
                            edges_add_back[(p, node, e)] = es[e]
            for s in graph1.successors(node):
                if s not in nodes:
                    es = graph1.get_edge_data(node, s)
                    if es is not None:
                        for e in es:
                            edges_add_back[(node, s, e)] = es[e]
        for node in nodes:
            graph1.remove_node(node)
        for node in graph2.nodes:
            graph1.add_nodes_from({node: graph2.nodes[node]}, **graph2.nodes[node])
        for edge in graph2.edges:
            graph1.add_edges_from({edge: graph2.edges[edge]}, **graph2.edges[edge])
        for edge in edges_add_back:
            if edge not in graph1.edges:
                graph1.add_edges_from({edge: edges_add_back[edge]}, **edges_add_back[edge])
        return graph1
    else:
        graph1.remove_edge(source, target, id)
        for node in graph2.nodes:
            if node != source and node != target:
                graph1.add_nodes_from({node: graph2.nodes[node]}, **graph2.nodes[node])
        for edge in graph2.edges:
            graph1.add_edges_from({edge: graph2.edges[edge]}, **graph2.edges[edge])
        return graph1


def segment_checkpoint_forward(segment):
    def custom_forward(*inputs):
        outputs = segment(*inputs)
        return outputs

    return custom_forward

def graph_forward(G, source, target, x, do_checkpoint=True):
    '''
    Do checkpoint forward with each vertex in G as gradient checkpoint or do regular forward with G
    :param G: networkx DAG
    :param source: source vertex key
    :param target: target vertex key
    :param x: input tensor
    :param do_checkpoint: whether to do regular forward or checkpoint forward
    :return: tuple (output tensor, source vertex id, edge id)
    '''

    tensor_dict = {source: (x, torch.tensor([-1.], requires_grad=True), torch.tensor([-1.], requires_grad=True))}
    queue = Queue()
    queue.put(source)
    while not queue.empty():
        vertex_key = queue.get()
        for target_vertex_id in G.successors(vertex_key):
            edges = G.get_edge_data(vertex_key, target_vertex_id)
            target_vertex = G.nodes[target_vertex_id]
            outputs = {}
            for id in edges:
                op = edges[id]['module']
                input, _, _ = tensor_dict[vertex_key]
                if do_checkpoint:
                    output = checkpoint(segment_checkpoint_forward(op), input)
                else:
                    output = op(input)

                if type(output) == tuple:
                    outputs.update(tuple_to_dict(output))
                else:
                    output = (output, torch.tensor([float(vertex_key)], requires_grad=True), torch.tensor([float(id)], requires_grad=True))
                    outputs.update(tuple_to_dict(output))
            transition = target_vertex.get('transition', None)
            if transition is None or (len([n for n in G.predecessors(target_vertex_id)]) == 1 and len(edges) == 1):
                tensor_dict[target_vertex_id] = outputs[list(outputs.keys())[0]]
                queue.put(target_vertex_id)
            else:
                # handle multi inputs
                transition_input_order = target_vertex['transition_input_order']
                num_input = len(transition_input_order)

                input_tuple = list(tensor_dict.get(target_vertex_id, ()))
                for key in outputs:
                    # a workaround
                    input_tuple.append(outputs[key][0])
                    input_tuple.append(outputs[key][1])
                    input_tuple.append(outputs[key][2])

                tensor_dict[target_vertex_id] = tuple(input_tuple)

                if len(input_tuple) == num_input * 3:
                    input_dict = tuple_to_dict(input_tuple)
                    inputs = [input_dict[i][0] for i in transition_input_order]
                    tensor_dict[target_vertex_id] = (transition(inputs), torch.tensor([-1.], requires_grad=True), torch.tensor([-1.], requires_grad=True))
                    queue.put(target_vertex_id)


    return tensor_dict[target]



class Segment(nn.Module):
    '''
    wrapper class for inference with DAG
    '''
    def __init__(self, G, source, target):
        super(Segment, self).__init__()
        self.G = G
        self.source = source
        self.target = target

    def forward(self, x):
        source = self.source
        target = self.target
        G = self.G
        return graph_forward(G, source, target, x, do_checkpoint=False)