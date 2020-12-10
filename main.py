import torch
from solver import ArbitrarySolver
from graph import graph_forward, set_graph_training
from utils import set_reproductibility
import time
import numpy as np
from tqdm import tqdm
from net.model_factory import model_factory, input_sizes

torch.backends.cudnn.enabled = True

def forward_check(net, graph, run_graph, source, target, device, input_size=(1,3,224,224)):
    inp = torch.rand(*input_size).to(device)
    net.eval()
    set_graph_training(graph, train=False)
    set_graph_training(run_graph, train=False)

    with torch.no_grad():
        ori_output = net(inp)
        graph_output, _, _ = graph_forward(graph, source, target, inp, do_checkpoint=False)
        run_graph_output, _, _ = graph_forward(run_graph, source, target, inp, do_checkpoint=True)

    max_graph_err = torch.max(torch.abs(graph_output - ori_output))
    if max_graph_err < 1e-05:
        print('Parsed graph forward check passed')
    else:
        print('Parsed graph forward check failed: Max Difference {}'.format(max_graph_err))

    max_run_graph_err = torch.max(torch.abs(run_graph_output - ori_output))
    if max_run_graph_err < 1e-05:
        print('Run graph forward check passed')
    else:
        print('Run graph forward check failed: Max Difference {}'.format(max_run_graph_err))


def backward_check(net, graph, run_graph, source, target, device, input_size=(1,3,224,224)):
    # todo: some parameters are not shared reference among net, graph and run_graph

    inp = torch.rand(*input_size).to(device)
    inp.requires_grad = True
    net.train()
    set_graph_training(graph, train=True)
    set_graph_training(run_graph, train=True)
    ori_output = net(inp)
    output_target = torch.rand(*ori_output.shape).to(device)
    loss = torch.sum(output_target - ori_output)
    loss.backward()
    ori_grad = [p.grad.clone() for p in net.parameters()]
    net.zero_grad()
    del ori_output, loss
    torch.cuda.empty_cache()

    graph_output, _, _ = graph_forward(graph, source, target, inp, do_checkpoint=False)
    loss = torch.sum(output_target - graph_output)
    loss.backward()
    graph_grad = [p.grad.clone() for p in net.parameters()]

    net.zero_grad()

    run_graph_output, _, _ = graph_forward(run_graph, source, target, inp, do_checkpoint=True)
    loss = torch.sum(output_target - run_graph_output)
    loss.backward()
    run_graph_grad = [p.grad.clone() for p in net.parameters()]

    max_graph_err = 0
    for g1, g2 in zip(ori_grad, graph_grad):
        rel_err = torch.max(torch.abs(g2 - g1)) / torch.norm(g1)
        if rel_err > max_graph_err:
            max_graph_err = rel_err

    if max_graph_err < 1e-05:
        print('Parsed graph backward check passed')
    else:
        print('Parsed graph backward check failed: Max Difference {}'.format(max_graph_err))

    max_run_graph_err = 0
    for g1, g2 in zip(ori_grad, run_graph_grad):
        rel_err = torch.max(torch.abs(g2 - g1)) / torch.norm(g1)
        if rel_err > max_run_graph_err:
            max_run_graph_err = rel_err

    if max_run_graph_err < 1e-05:
        print('Run graph backward check passed')
    else:
        print('Run graph backward check failed: Max Difference {}'.format(max_run_graph_err))

def forward_backward_net(net, device, input_size=(1,3,224,224), repeat=100, min_repeat=5):

    # do backward 1 time to get gradients counted
    input2 = torch.rand(*input_size, device=device)
    input2.requires_grad = True
    output2 = net(input2)
    loss = torch.sum(output2)
    loss.backward()
    # del input2, output2, loss

    # torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device)
    regular_start_memory = torch.cuda.max_memory_allocated(device)
    regular_times = []

    for i in tqdm(range(repeat)):
        start = time.time()
        input2 = torch.rand(*input_size, device=device)
        input2.requires_grad = True
        output2 = net(input2)
        loss = torch.sum(output2)
        loss.backward()
        end = time.time()
        regular_times.append(end - start)
        # del input2, output2, loss
    regular_peak_memory = torch.cuda.max_memory_allocated(device)
    # torch.cuda.empty_cache()
    regular_end_memory = torch.cuda.memory_allocated(device)
    regular_avg_time = np.mean(np.array(regular_times)[min_repeat:])

    return regular_start_memory, regular_end_memory, regular_peak_memory, regular_avg_time

def forward_backward_run_graph(run_graph, source, target, device, input_size=(1,3,224,224), repeat=100, min_repeat=5):
    # do backward 1 time to get gradients counted
    input1 = torch.rand(*input_size, device=device)
    input1.requires_grad = True
    output1, _, _ = graph_forward(run_graph, source, target, input1, do_checkpoint=True)
    loss = torch.sum(output1)
    loss.backward()
    # del input1, output1, loss
    # torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device)
    checkpoint_start_memory = torch.cuda.memory_allocated(device)
    checkpoint_times = []
    for i in tqdm(range(repeat)):
        start = time.time()
        input1 = torch.rand(*input_size, device=device)
        input1.requires_grad = True
        output1, _, _ = graph_forward(run_graph, source, target, input1, do_checkpoint=True)
        loss = torch.sum(output1)
        loss.backward()
        end = time.time()
        checkpoint_times.append(end - start)
        # del input1, output1, loss
    checkpoint_peak_memory = torch.cuda.max_memory_allocated(device)
    # torch.cuda.empty_cache()
    checkpoint_end_memory = torch.cuda.memory_allocated(device)
    checkpoint_avg_time = np.mean(np.array(checkpoint_times)[min_repeat:])

    return checkpoint_start_memory, checkpoint_end_memory, checkpoint_peak_memory, checkpoint_avg_time


def forward_backward_benchmark(net, run_graph, source, target, device, input_size=(1,3,224,224), repeat=100, min_repeat=5):
    assert repeat > min_repeat
    net.train()

    regular_start_memory, regular_end_memory, regular_peak_memory, regular_avg_time = forward_backward_net(net, device, input_size, repeat, min_repeat)
    checkpoint_start_memory, checkpoint_end_memory, checkpoint_peak_memory, checkpoint_avg_time = forward_backward_run_graph(run_graph, source, target, device, input_size, repeat, min_repeat)

    regular_pytorch_overhead = max(regular_start_memory, regular_end_memory)
    checkpoint_pytorch_overhead = max(checkpoint_start_memory, checkpoint_end_memory)

    regular_intermediate_tensors = regular_peak_memory - regular_pytorch_overhead
    checkpoint_intermediate_tensors = checkpoint_peak_memory - checkpoint_pytorch_overhead


    print('Average Iteration Time: Checkpointing {:.4f} s, Regular {:.4f} s, overhead {:.2f}%'.format(
        checkpoint_avg_time, regular_avg_time, (checkpoint_avg_time - regular_avg_time) * 100 / regular_avg_time))
    print('Average Peak Memory: Checkpointing {:.4f} MB, Regular {:.4f} MB, Memory Cut off {:.2f}%'.format(
        checkpoint_peak_memory / (1024**2), regular_peak_memory / (1024**2), (regular_peak_memory - checkpoint_peak_memory) * 100 / regular_peak_memory))
    print('Average Intermediate Tensors: Checkpointing {:.4f} MB, Regular {:.4f} MB, Memory Cut off {:.2f}%'.format(
        checkpoint_intermediate_tensors / (1024 ** 2), regular_intermediate_tensors / (1024 ** 2), (regular_intermediate_tensors - checkpoint_intermediate_tensors) * 100 / regular_intermediate_tensors))




def main(arch, device):
    set_reproductibility(2020)
    input_size = input_sizes[arch]
    print('Processing {}, Input size {}'.format(arch, input_size) + '-' * 20)
    net = model_factory[arch]().to(device)
    net.eval()
    inp = torch.rand(*input_size).to(device)
    G, source, target = net.parse_graph(inp)
    solver = ArbitrarySolver()

    start = time.time()
    run_graph, best_cost = solver.solve(G, source, target)
    end = time.time()
    print('Solving optimal gradient checkpointing takes {:.4f} s'.format(end - start))


    del inp
    torch.cuda.empty_cache()
    forward_check(net, G, run_graph, source, target, device, input_size=input_size)
    backward_check(net, G, run_graph, source, target, device, input_size=input_size)
    forward_backward_benchmark(net, run_graph, source, target, device, input_size=input_size, repeat=100, min_repeat=10)
    del net, G, run_graph
    torch.cuda.empty_cache()

def run_all():
    ### alexnet, vgg, inception backward check will fail because of drop out
    device = torch.device('cuda:0')
    for arch in model_factory:
        main(arch, device)



if __name__ == '__main__':
    run_all()