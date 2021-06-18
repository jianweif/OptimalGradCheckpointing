from torch import nn as nn
from torch.utils.checkpoint import checkpoint
from queue import Queue
import networkx as nx
import torch
import torch.nn.functional as F
from net.layer import TupleConstruct, TupleIndexing, Mul2, Add2, BasicIdentity, Cat, ListConstruct, Flatten, View, FunctionWrapperV2
from copy import deepcopy

# todo: get shapes of all the tensors when tracing

Basic_ops = (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d,
             nn.AdaptiveMaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
             nn.Bilinear, nn.CELU, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
             nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d, nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.ELU, nn.Embedding,
             nn.EmbeddingBag, nn.FeatureAlphaDropout, nn.FractionalMaxPool2d, nn.FractionalMaxPool3d, nn.GELU, nn.GLU, nn.GroupNorm,
             nn.GRU, nn.GRUCell, nn.Hardtanh, nn.Identity, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
             nn.LayerNorm, nn.LeakyReLU, nn.Linear, nn.LocalResponseNorm, nn.LogSigmoid, nn.LPPool1d, nn.LPPool2d, nn.LeakyReLU,
             nn.LogSoftmax, nn.LSTM, nn.LSTMCell, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.MaxUnpool1d, nn.MaxUnpool2d,
             nn.MaxUnpool3d, nn.MultiheadAttention, nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU, nn.ReflectionPad1d, nn.ReflectionPad2d,
             nn.ReplicationPad1d, nn.ReplicationPad2d, nn.ReplicationPad3d, nn.SELU, nn.Sigmoid, nn.Softmax, nn.Softmax2d,
             nn.Softmin, nn.Softplus, nn.Softshrink, nn.Softshrink, nn.Tanh, nn.Tanhshrink, nn.Upsample, nn.UpsamplingBilinear2d,
             nn.UpsamplingNearest2d, nn.ZeroPad2d)
Multi_input_ops = (TupleConstruct, ListConstruct)


def parse_computation_graph(module, inputs):
    '''
    :param module: nn.module to parse for computation graph
    :param input: torch.Tensor, example input tensor
    :return: nx.MultiDiGraph
    '''

    computation_graph, input_node_ids, output_node_ids = parse_raw_computation_graph_from_jit(module, inputs)
    computation_graph = optimize_computation_graph(computation_graph, input_node_ids, output_node_ids)

    sources, targets = get_source_target(computation_graph)
    if len(sources) > 1 or len(targets) > 1:
        raise Exception("Currently not supporting multi input or output graph, we are working on supporting it")
    source, target = sources[0], targets[0]
    with torch.no_grad():
        tmp_parsed_segment = Segment(computation_graph, sources[0], targets[0], do_checkpoint=False, record_tensor_cost=True)
        output = tmp_parsed_segment.forward(inputs[0])
    return computation_graph, source, target

def parse_raw_computation_graph_from_jit(module, inputs):
    '''
    :param module: nn.module to parse for computation graph
    :param input: torch.Tensor, example input tensor
    :return: nx.MultiDiGraph
    '''
    add_input_tensor_hook_recursively(module)
    output = module.forward(*inputs)
    remove_input_tensor_hook_recursively(module)
    computation_graph, _, input_node_ids, output_node_ids = build_computation_graph_recursively(module, inputs, inputs_nodes_ids=None, outputs_nodes_ids=None, cur_node_idx=None)
    clean_up_input_tensor_recursively(module)
    return computation_graph, input_node_ids, output_node_ids

def classify_node_type(node_type):
    # todo: may need to refine
    if node_type[0] == '(' and node_type[-1] == ')':
        # parse tuple
        return 'Tuple'
    elif node_type == 'Tensor[]':
        return 'List'
    elif node_type in ['int', 'float', 'bool', 'int[]', 'float[]', 'bool[]', 'None']:
        return node_type
    elif node_type == 'Tensor' or 'Float' in node_type or 'Long' in node_type:
        # todo: may need to add more dtype
        return 'Tensor'
    else:
        return 'Module'

def parse_node_op(node_op):
    splits = node_op.split('(')
    op_def = '('.join(splits[:-1])
    op_args = splits[-1].strip(')')
    if len(op_args) == 0:
        op_args = []
    else:
        op_args = op_args.split(', ')
    return {'op_def': op_def, 'op_args': op_args}

def retrieve_constant_value(local_graph_dict, node_class, node_op):
    if node_class in ['int', 'float', 'bool']:
        op_def = node_op['op_def']
        if 'prim::Constant' in op_def:
            dtype = eval(node_class)
            value_str = op_def.split('[')[-1].split(']')[0].replace('value=', '')
            return (dtype)(value_str)
        elif op_def == 'aten::size':
            op_args = node_op['op_args']
            tensor_node, index_node = op_args
            tensor_shape = local_graph_dict[tensor_node]['shape']
            index = local_graph_dict[index_node]['value']
            return tensor_shape[index]
        elif op_def == 'prim::NumToTensor':
            # todo: we are handling tensor from numToTensor as constant and directly treat its value as int/float/bool, this might have risk
            op_args = node_op['op_args']
            return local_graph_dict[op_args[0]]['value']
        elif op_def == 'aten::Int':
            op_args = node_op['op_args']
            return local_graph_dict[op_args[0]]['value']
        else:
            raise NotImplementedError
    elif node_class == 'None':
        return None
    elif node_class in ['int[]', 'float[]', 'bool[]']:
        op_args = node_op['op_args']
        return [local_graph_dict[n]['value'] for n in op_args]
    else:
        raise NotImplementedError

def parse_input_node_str(node_str):
    # remove comment
    node_str = node_str.split(' #')[0]
    node_groups = node_str.split(', %')
    for i in range(1, len(node_groups)):
        # add back )
        node_groups[i] = '%' + node_groups[i]
    node_dict = {}
    for node_group in node_groups:
        if ' = ' in node_group:
            node_def, node_op = node_group.split(' = ')
        else:
            node_def = node_group
        node_name, node_type = node_def.split(' : ')
        node_class = classify_node_type(node_type)
        if node_name not in node_dict:
            node_dict[node_name] = {'node_class': node_class, 'node_op': None, 'output_id': None}
    return node_dict


def parse_node_str(node_str):
    # remove comment
    node_str = node_str.split(' #')[0]
    op_groups = node_str.split('), %')
    for i in range(len(op_groups) - 1):
        # add back )
        op_groups[i] += ')'
    for i in range(1, len(op_groups)):
        # add back )
        op_groups[i] = '%' + op_groups[i]

    node_dict = {}
    for op_group in op_groups:
        node_group, node_op = op_group.split(' = ')
        node_op = parse_node_op(node_op)
        node_defs = node_group.split(', %')
        for i in range(1, len(node_defs)):
            # add back %
            node_defs[i] = '%' + node_defs[i]
        for i, node_def in enumerate(node_defs):
            node_name, node_type = node_def.split(' : ')
            node_class = classify_node_type(node_type)
            if node_name not in node_dict:
                node_dict[node_name] = {'node_class': node_class, 'node_op': node_op, 'output_id': i}
                if node_class == 'Tensor' and '(' in node_type and ')' in node_type:
                    # try to get shape
                    shape_str = node_type.split('(')[-1].split(')')[0]
                    if ', ' in shape_str:
                        shape = [int(s) for s in shape_str.split(', ')]
                        node_dict[node_name]['shape'] = shape
                    else:
                        node_dict[node_name]['shape'] = []

                # if node_class in ['int', 'float', 'bool']:
                #     value = retrieve_constant_value(node_class, node_op)
                #     node_dict[node_name]['value'] = value

    '''
    # remove comment
    node_str = node_str.split(' #')[0]
    splits = node_str.split(', %')
    for i in range(1, len(splits)):
        # add back %
        splits[i] = '%' + splits[i]
    node_dict = {}
    queue = []
    for s in splits:
        if ' = ' in s:
            node_def, node_op = s.split(' = ')
            node_name, node_type = node_def.split(' : ')
            node_class = classify_node_type(node_type)
            node_dict[node_name] = {'node_class': node_class, 'node_op': None, 'output_id': None}
            queue.append(node_name)
            node_op = parse_node_op(node_op)
            for i, queued_node_name in enumerate(queue):
                node_dict[queued_node_name]['node_op'] = node_op
                node_dict[queued_node_name]['output_id'] = i
            queue = []
        else:
            node_name, node_type = s.split(' : ')
            node_class = classify_node_type(node_type)
            node_dict[node_name] = {'node_class': node_class, 'node_op': None, 'output_id': None}
            queue.append(node_name)
    '''
    return node_dict

'''def parse_node_def(node_def):
    splits = node_def.split(' = ')
    if len(splits) == 1:
        node_type = splits[0]
        node_class = classify_node_type(node_type)
        return {'node_class': node_class, 'node_op': None}
    elif len(splits) == 2:
        node_type, node_op = splits
        node_class = classify_node_type(node_type)
        node_op = parse_node_op(node_op)
        return {'node_class': node_class, 'node_op': node_op}
    else:
        raise NotImplementedError

def parse_inputs(graph_inputs):
    local_graph_dict = {}
    for i in graph_inputs:
        input_str = str(i)
        node_strs = input_str.split(', ')
        for node_str in node_strs:
            # remove comment
            node_str = node_str.split(' #')[0]
            node_name, node_def = node_str.split(' : ')
            if node_name in local_graph_dict:
                continue
            # parse node_def
            local_graph_dict[node_name] = parse_node_def(node_def)
    return local_graph_dict

def parse_nodes(graph_nodes, local_graph_dict={}):
    for n in graph_nodes:
        node_str = str(n)
        # remove comment
        node_str = node_str.split(' #')[0]
        node_name, node_def = node_str.split(' : ')
        if node_name in local_graph_dict:
            continue
        # parse node_def
        local_graph_dict[node_name] = parse_node_def(node_def)
    return local_graph_dict

def parse_outputs(graph_outputs, local_graph_dict={}):
    for o in graph_outputs:
        node_str = str(o)
        # remove comment
        node_str = node_str.split(' #')[0]
        node_name, node_def = node_str.split(' : ')
        if node_name in local_graph_dict:
            continue
        # parse node_def
        local_graph_dict[node_name] = parse_node_def(node_def)
    return local_graph_dict'''

def get_python_module(local_graph_dict, module, node_name):
    node_info = local_graph_dict[node_name]
    node_class, node_op = node_info['node_class'], node_info['node_op']
    if node_class == 'Module':
        if node_op == None or node_name == '%self.1':
            local_graph_dict[node_name]['python_module'] = module
            return local_graph_dict
        op_def, op_args = node_op['op_def'], node_op['op_args']
        if len(op_args) == 1:
            if 'prim::GetAttr' in op_def:
                parent_node_name = op_args[0]
                if 'python_module' not in local_graph_dict[parent_node_name]:
                    raise Exception("python_module not defined for {}".format(parent_node_name))
                parent_module = local_graph_dict[parent_node_name]['python_module']
                attr_name = op_def.split('[')[-1].split(']')[0].replace("name=", '').strip("\"")
                local_graph_dict[node_name]['python_module'] = getattr(parent_module, attr_name)
            else:
                raise Exception(
                    "op_def {} conversion to python not implemented, please raise an issue on github".format(op_def))
        else:
            raise Exception("Module {} not recognized, op def {}, op args {}".format(node_name, op_def, op_args))
    return local_graph_dict

def get_python_modules(local_graph_dict, module):
    # translate all the modules in local_graph_dict to python modules
    for node_name in local_graph_dict:
        node_info = local_graph_dict[node_name]
        node_class, node_op = node_info['node_class'], node_info['node_op']
        if node_class == 'Module':
            if node_op == None or node_name == '%self.1':
                local_graph_dict[node_name]['python_module'] = module
                continue
            op_def, op_args = node_op['op_def'], node_op['op_args']
            if len(op_args) == 1:
                if 'prim::GetAttr' in op_def:
                    parent_node_name = op_args[0]
                    if 'python_module' not in local_graph_dict[parent_node_name]:
                        raise Exception("python_module not defined for {}".format(parent_node_name))
                    parent_module = local_graph_dict[parent_node_name]['python_module']
                    attr_name = op_def.split('[')[-1].split(']')[0].replace("name=", '').strip("\"")
                    local_graph_dict[node_name]['python_module'] = getattr(parent_module, attr_name)
                else:
                    raise Exception("op_def {} conversion to python not implemented, please raise an issue on github".format(op_def))
            else:
                raise Exception("Module {} not recognized, op def {}, op args {}".format(node_name, op_def, op_args))
    return local_graph_dict

def get_python_module_from_node_op(local_graph_dict, node_op, output_id, local_node_mapping):
    # todo: add drop out, concat, relu, ...
    # todo: add code for functions from torch.nn.functional
    # todo: add code for constant add, mul, ...
    op_def, op_args, op_output_id = node_op['op_def'], node_op['op_args'], output_id
    if 'prim::CallMethod' in op_def:
        module_node = op_args[0]
        python_module = local_graph_dict[module_node]['python_module']
        if isinstance(python_module, Basic_ops):
            basic_op = True
        else:
            basic_op = False
        input_nodes = op_args[1:]
        input_node_ids = [local_node_mapping[n] for n in input_nodes]
        return {'python_module': python_module, 'input_node_ids': input_node_ids, 'basic_op': basic_op}
    elif 'prim::TupleConstruct' in op_def:
        python_module = TupleConstruct()
        basic_op = True
        input_nodes = op_args
        input_node_ids = [local_node_mapping[n] for n in input_nodes]
        return {'python_module': python_module, 'input_node_ids': input_node_ids, 'basic_op': basic_op}
    elif 'prim::TupleUnpack' in op_def:
        python_module = TupleIndexing(index=op_output_id)
        basic_op = True
        input_nodes = op_args
        input_node_ids = [local_node_mapping[n] for n in input_nodes]
        return {'python_module': python_module, 'input_node_ids': input_node_ids, 'basic_op': basic_op}
    elif 'prim::ListConstruct' in op_def:
        python_module = ListConstruct()
        basic_op = True
        input_nodes = op_args
        input_node_ids = [local_node_mapping[n] for n in input_nodes]
        return {'python_module': python_module, 'input_node_ids': input_node_ids, 'basic_op': basic_op}
    elif 'aten::mul' in op_def:
        python_module = Mul2()
        basic_op = True
        input_nodes = op_args
        input_node_ids = [local_node_mapping[n] for n in input_nodes]
        return {'python_module': python_module, 'input_node_ids': input_node_ids, 'basic_op': basic_op}
    elif 'aten::add' in op_def:
        python_module = Add2()
        basic_op = True
        input_nodes = op_args[:-1]
        #todo: not sure what constant do here
        constant = local_graph_dict[op_args[-1]]['value']
        input_node_ids = [local_node_mapping[n] for n in input_nodes]
        return {'python_module': python_module, 'input_node_ids': input_node_ids, 'basic_op': basic_op}
    elif 'aten::cat' in op_def:
        assert len(op_args) == 2
        basic_op = True
        input_nodes = [op_args[0]]
        dim = local_graph_dict[op_args[1]]['value']
        python_module = Cat(dim=dim)
        input_node_ids = [local_node_mapping[n] for n in input_nodes]
        return {'python_module': python_module, 'input_node_ids': input_node_ids, 'basic_op': basic_op}
    elif 'aten::flatten' in op_def:
        assert len(op_args) == 3
        basic_op = True
        input_nodes = [op_args[0]]
        # todo: not sure what this constant means
        constant = local_graph_dict[op_args[1]]['value']
        dim = local_graph_dict[op_args[2]]['value']
        python_module = Flatten(dim=dim)
        input_node_ids = [local_node_mapping[n] for n in input_nodes]
        return {'python_module': python_module, 'input_node_ids': input_node_ids, 'basic_op': basic_op}
    elif 'aten::relu_' in op_def:
        assert len(op_args) == 1
        python_module = nn.ReLU(inplace=True)
        basic_op = True
        input_node_ids = [local_node_mapping[n] for n in op_args]
        return {'python_module': python_module, 'input_node_ids': input_node_ids, 'basic_op': basic_op}
    elif 'aten::view' in op_def:
        assert len(op_args) == 2
        basic_op = True
        input_nodes = [op_args[0]]
        shape = local_graph_dict[op_args[1]]['value']
        python_module = View(shape=shape)
        input_node_ids = [local_node_mapping[n] for n in input_nodes]
        return {'python_module': python_module, 'input_node_ids': input_node_ids, 'basic_op': basic_op}
    elif op_def in ['aten::max_pool2d', 'aten::adaptive_avg_pool2d', 'aten::avg_pool2d', 'aten::dropout']:
        func_name = op_def.replace('aten::', '')
        func = getattr(F, func_name)
        basic_op = True
        input_nodes = [op_args[0]]
        args = [local_graph_dict[n]['value'] for n in op_args[1:]]
        python_module = FunctionWrapperV2(run_func=func, run_args=args)
        input_node_ids = [local_node_mapping[n] for n in input_nodes]
        return {'python_module': python_module, 'input_node_ids': input_node_ids, 'basic_op': basic_op}
    else:
        raise Exception("op_def {} conversion to python not implemented, please raise an issue on github".format(op_def))

def merge_dict_list(dict_list):
    new_dict = {}
    for d in dict_list:
        for key in d:
            if key not in new_dict:
                new_dict[key] = d[key]
    return new_dict

def build_computation_graph_recursively(module, inputs, inputs_nodes_ids=None, outputs_nodes_ids=None, cur_node_idx=None):
    device = inputs[0].device
    if cur_node_idx is None:
        cur_node_idx = 0
    with torch.no_grad():
        traced = torch.jit.trace(module.forward, tuple(inputs))
    del inputs
    traced_graph = traced.graph
    graph_inputs = [str(i.node()).strip('\n') for i in traced_graph.inputs()]
    graph_nodes = [str(n).strip('\n') for n in traced_graph.nodes()]
    graph_outputs = [str(o.node()).strip('\n') for o in traced_graph.outputs()]

    input_node_dicts = [parse_input_node_str(i) for i in graph_inputs]
    internal_node_dicts = [parse_node_str(n) for n in graph_nodes]
    output_node_dicts = [parse_node_str(o) for o in graph_outputs]
    node_dicts = input_node_dicts + internal_node_dicts + output_node_dicts

    local_graph_dict = merge_dict_list(node_dicts)
    inputs_dict = merge_dict_list(input_node_dicts)
    outputs_dict = merge_dict_list(output_node_dicts)

    # local_graph_dict = parse_inputs(graph_inputs)
    # local_graph_dict = parse_nodes(graph_nodes, local_graph_dict)
    # local_graph_dict = parse_outputs(graph_outputs, local_graph_dict)

    local_graph_dict = get_python_modules(local_graph_dict, module)

    local_node_mapping = {}
    if inputs_nodes_ids is not None:
        input_node_names = [n for n in inputs_dict if inputs_dict[n]['node_class'] != 'Module']
        assert len(input_node_names) == len(inputs_nodes_ids)
        for input_node_name, input_node_id in zip(input_node_names, inputs_nodes_ids):
            local_node_mapping[input_node_name] = input_node_id
    else:
        # allocate input tensors first
        inputs_nodes_ids = []
        input_node_names = [n for n in inputs_dict if inputs_dict[n]['node_class'] != 'Module']
        for input_node_name in input_node_names:
            local_node_mapping[input_node_name] = cur_node_idx
            inputs_nodes_ids.append(cur_node_idx)
            cur_node_idx += 1
    if outputs_nodes_ids is not None:
        output_node_names = [n for n in outputs_dict if outputs_dict[n]['node_class'] != 'Module']
        assert len(output_node_names) == len(outputs_nodes_ids)
        for output_node_name, output_node_id in zip(output_node_names, outputs_nodes_ids):
            local_node_mapping[output_node_name] = output_node_id
    else:
        # allocate output tensors
        outputs_nodes_ids = []
        output_node_names = [n for n in outputs_dict if outputs_dict[n]['node_class'] != 'Module']
        for output_node_name in output_node_names:
            local_node_mapping[output_node_name] = cur_node_idx
            outputs_nodes_ids.append(cur_node_idx)
            cur_node_idx += 1

    graph = nx.MultiDiGraph()
    for node_name in local_graph_dict:
        node_info = local_graph_dict[node_name]
        node_class, node_op, node_output_id = node_info['node_class'], node_info['node_op'], node_info['output_id']
        # todo: rewrite node_class, a workaround, sometimes LongTensor will be created from int
        if node_op != None and node_op['op_def'] == 'prim::NumToTensor':
            node_class = 'int'
        if node_class in ['Tensor', 'Tuple', 'List']:
            if node_name not in local_node_mapping:
                # allocate node id
                node_idx = cur_node_idx
                local_node_mapping[node_name] = node_idx
                cur_node_idx += 1
            else:
                # use existing node id
                node_idx = local_node_mapping[node_name]

            graph.add_node(node_idx)
            if node_op != None:
                op_def, op_args = node_op['op_def'], node_op['op_args']
                if len(op_args) > 0:
                    op_input_node_names = op_args
                    # run a sanity check
                    for input_node_name in op_input_node_names:
                        if input_node_name not in local_node_mapping:
                            raise Exception("{} is input node for op {}, but not recorded by local_node_mapping".format(input_node_name, op_def))

                    python_module_dict = get_python_module_from_node_op(local_graph_dict, node_op, node_output_id, local_node_mapping)
                    python_module, node_input_ids, basic_op = python_module_dict['python_module'], python_module_dict['input_node_ids'], python_module_dict['basic_op']
                    if basic_op:
                        if len(node_input_ids) > 1:
                            # multi-input op
                            transition_op = python_module
                            transition_input_order = []
                            for node_input_id in node_input_ids:
                                identity = BasicIdentity()
                                graph.add_edge(node_input_id, node_idx, cost=0, module=identity)
                                transition_input_order.append((node_input_id, 0))
                            graph.nodes[node_idx]['transition'] = transition_op
                            graph.nodes[node_idx]['transition_input_order'] = transition_input_order
                        elif len(node_input_ids) == 1:
                            node_input_id = node_input_ids[0]
                            graph.add_edge(node_input_id, node_idx, cost=0, module=python_module)
                        else:
                            raise Exception("op_def {} has no input nodes".format(op_def))
                    else:
                        # construct computation graph recursively
                        node_inputs = python_module.__input_tensor__
                        subgraph, cur_node_idx, _, _ = build_computation_graph_recursively(python_module, node_inputs,
                            inputs_nodes_ids=node_input_ids, outputs_nodes_ids=[node_idx], cur_node_idx=cur_node_idx)
                        del python_module.__input_tensor__
                        # merge subgraph in graph
                        for node in subgraph.nodes:
                            if node not in graph.nodes:
                                graph.add_nodes_from({node: subgraph.nodes[node]}, **subgraph.nodes[node])
                            else:
                                # add attributes
                                for key in subgraph.nodes[node]:
                                    graph.nodes[node][key] = subgraph.nodes[node][key]
                        for edge in subgraph.edges:
                            graph.add_edges_from({edge: subgraph.edges[edge]}, **subgraph.edges[edge])
                elif 'prim::Param' in op_def:
                    pass
                else:
                    raise Exception("Unrecognized op_def {} with empty input args".format(op_def))
        elif node_class in ['int[]', 'float[]', 'bool[]', 'int', 'float', 'bool', 'None']:
            # implement here to retreive constant list
            local_node_mapping[node_name] = None
            value = retrieve_constant_value(local_graph_dict, node_class, node_op)
            local_graph_dict[node_name]['value'] = value
        elif node_class == 'Module':
            local_node_mapping[node_name] = None
            local_graph_dict = get_python_module(local_graph_dict, module, node_name)
        else:
            local_node_mapping[node_name] = None

    return graph, cur_node_idx, inputs_nodes_ids, outputs_nodes_ids

def optimize_computation_graph(G, input_node_ids, output_node_ids):
    G = merge_tuple_op(G)
    G = trim_unused_nodes(G, input_node_ids, output_node_ids)
    G = rewrite_multi_input_op(G)
    G = merge_inplace_op(G)
    return G

def trim_unused_nodes(graph, input_node_ids, output_node_ids):
    '''
    remove unused nodes (no incoming edge or no outgoing edge)
    :param graph: nx.MultiDiGraph
    :param input_node_ids: list of input node indices
    :param output_node_ids: list of output node indices
    :return:
    '''
    edges = [e for e in graph.edges()]
    nodes = [n for n in graph.nodes()]
    source_set = set([e[0] for e in edges])
    target_set = set([e[1] for e in edges])
    used_node_set = source_set.intersection(target_set)
    for input_node_id in input_node_ids:
        used_node_set.add(input_node_id)
    for output_node_id in output_node_ids:
        used_node_set.add(output_node_id)
    unused_node_set = set(nodes).difference(used_node_set)
    if len(unused_node_set) == 0:
        return graph
    else:
        for node in unused_node_set:
            graph.remove_node(node)
        graph = trim_unused_nodes(graph, input_node_ids, output_node_ids)
        return graph

def merge_tuple_op(graph):
    '''
    remove tuple construct and tuple indexing edges and merge nodes
    :param graph: nx.MultiDiGraph
    :return:
    '''
    tuple_node_ids = [n for n in graph.nodes if 'transition' in graph.nodes[n] and isinstance(graph.nodes[n]['transition'], TupleConstruct)]
    for tuple_node_id in tuple_node_ids:
        input_edges = graph.nodes[tuple_node_id]['transition_input_order']
        output_edges = [None for _ in input_edges]
        for e in graph.edges:
            s, t, id = e
            op = graph.edges[e]['module']
            if s == tuple_node_id:
                output_edges[op.index] = (t, id)
        merge_flag = True
        for output_edge in output_edges:
            output_op = graph.edges[(tuple_node_id, output_edge[0], output_edge[1])]['module']
            if not isinstance(output_op, TupleIndexing):
                merge_flag = False
                break
        if not merge_flag:
            continue
        # reroute the edges, and merge nodes before and after tuple
        graph.remove_node(tuple_node_id)
        for input_edge, output_edge in zip(input_edges, output_edges):
            input_node_id, output_node_id = input_edge[0], output_edge[0]
            # merge output node into input node
            for edge in graph.out_edges(output_node_id):
                multi_edges = graph.get_edge_data(edge[0], edge[1])
                for id in multi_edges:
                    edge_key = (edge[0], edge[1], id)
                    new_edge_key = (input_node_id, edge[1], id)
                    graph.add_edges_from({new_edge_key: graph.edges[edge_key]}, **graph.edges[edge_key])
                # rewrite transition_input_order
                if 'transition_input_order' in graph.nodes[edge[1]]:
                    for i, (trans_s, trans_id) in enumerate(graph.nodes[edge[1]]['transition_input_order']):
                        if trans_s == output_node_id:
                            graph.nodes[edge[1]]['transition_input_order'][i] = (input_node_id, graph.nodes[edge[1]]['transition_input_order'][i][1])
            graph.remove_node(output_node_id)
    return graph

def merge_inplace_op(graph):
    '''
    merge inplace operation such as nn.ReLU(inplace=True) into previous operations
    :param graph: nx.MultiDiGraph
    :return:
    '''
    inplace_edges = []
    for e in graph.edges:
        op = graph.edges[e]['module']
        if hasattr(op, 'inplace') and getattr(op, 'inplace'):
            inplace_edges.append(e)
    for e in inplace_edges:
        s, t, id = e
        inplace_op = graph.edges[e]['module']
        if 'transition' in graph.nodes[s]:
            # if the previous op is a multi input op (transition op) then merge into transition op
            graph.nodes[s]['transition'] = nn.Sequential(graph.nodes[s]['transition'], inplace_op)
        else:
            # merge inplace op into previous op
            for edge in graph.in_edges(s):
                multi_edges = graph.get_edge_data(edge[0], edge[1])
                for id in multi_edges:
                    edge_key = (edge[0], edge[1], id)
                    edge_op = graph.edges[edge_key]['module']
                    graph.edges[edge_key]['module'] = nn.Sequential(edge_op, inplace_op)
        # reroute outgoing edges
        for edge in graph.out_edges(t):
            multi_edges = graph.get_edge_data(edge[0], edge[1])
            for id in multi_edges:
                edge_key = (edge[0], edge[1], id)
                new_edge_key = (s, edge[1], id)
                graph.add_edges_from({new_edge_key: graph.edges[edge_key]}, **graph.edges[edge_key])
            # rewrite transition_input_order
            if 'transition_input_order' in graph.nodes[edge[1]]:
                for i, (trans_s, trans_id) in enumerate(graph.nodes[edge[1]]['transition_input_order']):
                    if trans_s == t:
                        graph.nodes[edge[1]]['transition_input_order'][i] = (s, graph.nodes[edge[1]]['transition_input_order'][i][1])
        graph.remove_node(t)
    return graph

def rewrite_multi_input_op(graph):
    multi_input_node_ids = [n for n in graph.nodes if 'transition' in graph.nodes[n] and isinstance(graph.nodes[n]['transition'], Multi_input_ops)]
    for multi_input_node_id in multi_input_node_ids:
        input_edges = graph.nodes[multi_input_node_id]['transition_input_order']

        for edge in graph.out_edges(multi_input_node_id):
            multi_edges = graph.get_edge_data(edge[0], edge[1])
            if len(multi_edges) > 1:
                raise Exception("More than 1 edges exist between 2 nodes when optimizing the graph")
            for id in multi_edges:
                edge_key = (edge[0], edge[1], id)
                op = graph.edges[edge_key]['module']
                graph.nodes[edge[1]]['transition'] = op
                graph.nodes[edge[1]]['transition_input_order'] = deepcopy(input_edges)
                for (tran_s, trans_id) in input_edges:
                    graph.add_edges_from({(tran_s, edge[1], trans_id): graph.edges[(tran_s, multi_input_node_id, trans_id)]},
                                         **graph.edges[(tran_s, multi_input_node_id, trans_id)])

        graph.remove_node(multi_input_node_id)

    return graph


def get_source_target(graph):
    edges = [e for e in graph.edges()]
    s_set = set([e[0] for e in edges])
    t_set = set([e[1] for e in edges])
    intermediate_node_set = s_set.intersection(t_set)
    source_set = s_set.difference(intermediate_node_set)
    target_set = t_set.difference(intermediate_node_set)
    return list(source_set), list(target_set)


def add_input_tensor_hook_recursively(module):
    if isinstance(module, Basic_ops):
        # handle = module.register_forward_hook(input_tensor_hook)
        # module.__hook_handle__ = handle
        pass
    else:
        handle = module.register_forward_hook(input_tensor_hook)
        module.__hook_handle__ = handle
        for name, sub_module in module._modules.items():
            add_input_tensor_hook_recursively(sub_module)


def input_tensor_hook(module, input, output):
    # module.__input_shape__ = [i.shape for i in input]
    module.__input_tensor__ = input

def remove_input_tensor_hook_recursively(module):
    if isinstance(module, Basic_ops):
        # module.__hook_handle__.remove()
        # del module.__hook_handle__
        pass
    else:
        module.__hook_handle__.remove()
        del module.__hook_handle__
        for name, sub_module in module._modules.items():
            remove_input_tensor_hook_recursively(sub_module)

def clean_up_input_tensor_recursively(module):
    if isinstance(module, Basic_ops):
        if hasattr(module, '__input_tensor__'):
            del module.__input_tensor__
    else:
        if hasattr(module, '__input_tensor__'):
            del module.__input_tensor__
        for name, sub_module in module._modules.items():
            clean_up_input_tensor_recursively(sub_module)


def tuple_to_dict(t):
    l = list(t)
    num = len(l) // 3
    d = {}
    for i in range(num):
        tensor, s, ind = t[i * 3], t[i * 3 + 1], t[i * 3 + 2]
        d[(int(s), int(ind))] = tensor
    return d

def dict_to_tuple(d):
    l = []
    for (s, ind) in d:
        tensor = d[(s, ind)]
        l.append(tensor)
        # has to use float otherwise throw requires_grad error
        l.append(torch.tensor([float(s)], requires_grad=True))
        l.append(torch.tensor([float(ind)], requires_grad=True))
    return tuple(l)

def set_segment_training(segment, train=True):
    set_graph_training(segment.G, train=train)


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

# NOTE: checkpoint autograd.function doesn't allow dictionary output, so have to use tensor to hold vertex id
def graph_forward(x, G=None, source=None, target=None, successors_dict=None, predecessors_dict=None, edges_dict=None, do_checkpoint=True, record_tensor_cost=False):
    '''
    Do checkpoint forward with each vertex in G as gradient checkpoint or do regular forward with G
    :param G: networkx DAG
    :param source: source vertex key
    :param target: target vertex key
    :param x: input tensor
    :param do_checkpoint: whether to do regular forward or checkpoint forward
    :param record_tensor_cost: whether to record the tensor cost during execution and store in G
    :return:
    '''

    tensor_dict = {source: x}
    queue = Queue()
    queue.put(source)
    while not queue.empty():
        vertex_key = queue.get()
        for target_vertex_id in successors_dict[vertex_key]:
            edges = edges_dict[(vertex_key, target_vertex_id)]
            target_vertex = G.nodes[target_vertex_id]
            outputs = {}
            for id in edges:
                op = edges[id]['module']
                input = tensor_dict[vertex_key]
                if do_checkpoint:
                    output = checkpoint(segment_checkpoint_forward(op), input)
                else:
                    output = op(input)

                if type(output) == tuple:
                    output = tuple_to_dict(output)
                    for key in output:
                        outputs[key] = output[key]
                else:
                    outputs[(vertex_key, id)] = output


            transition = target_vertex.get('transition', None)
            if transition is None:
                tensor_dict[target_vertex_id] = outputs[list(outputs.keys())[0]]
                queue.put(target_vertex_id)
            else:
                # handle multi inputs
                transition_input_order = target_vertex['transition_input_order']
                num_input = len(transition_input_order)

                inputs_for_transit = tensor_dict.get(target_vertex_id, {})
                for key in outputs:
                    inputs_for_transit[key] = outputs[key]
                if len(inputs_for_transit) == num_input:
                    inputs = [inputs_for_transit[i] for i in transition_input_order]
                    tensor_dict[target_vertex_id] = transition(inputs)
                    queue.put(target_vertex_id)
                else:
                    tensor_dict[target_vertex_id] = inputs_for_transit
    if record_tensor_cost:
        for node in tensor_dict:
            if type(tensor_dict[node]) == dict:
                pass
            else:
                node_cost = tensor_dict[node].numel()
                G.nodes[node]['cost'] = node_cost

    if type(tensor_dict[target]) == dict:
        return dict_to_tuple(tensor_dict[target])
    else:
        return tensor_dict[target]


class Segment(nn.Module):
    '''
    wrapper class for inference with DAG
    '''
    def __init__(self, G, source, target, do_checkpoint=False, record_tensor_cost=False):
        super(Segment, self).__init__()
        self.G = G
        self.source = source
        self.target = target
        self.info_dict = self.prepare_for_forward(G, source, target, do_checkpoint, record_tensor_cost)

    def prepare_for_forward(self, G, source, target, do_checkpoint, record_tensor_cost):
        info_dict = {'G': G, 'source': source, 'target': target}
        successors_dict, predecessors_dict, edges_dict = {}, {}, {}
        for v in G.nodes:
            predecessors_dict[v] = [n for n in G.predecessors(v)]
            successors_dict[v] = [n for n in G.successors(v)]
        for key in G.edges:
            e = G.edges[key]
            start, end, id = key
            if (start, end) not in edges_dict:
                edges_dict[(start, end)] = {}
            edges_dict[(start, end)][id] = e
        info_dict.update(successors_dict=successors_dict, predecessors_dict=predecessors_dict, edges_dict=edges_dict,
                         do_checkpoint=do_checkpoint, record_tensor_cost=record_tensor_cost)
        return info_dict


    def forward(self, x):
        return graph_forward(x, **self.info_dict)