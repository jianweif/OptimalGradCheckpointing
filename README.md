# Optimal Gradient Checkpoint Search for Arbitrary Computation Graphs
This is the official implementation of the paper: 

**Jianwei Feng and Dong Huang**, ***Optimal Gradient Checkpoint Search for Arbitrary Computation Graphs***, **CVPR, 2021 (Oral)**. [ArXiv version](https://arxiv.org/abs/1808.00079)

### Citation: 

```bash
@inproceedings{fenghuang2021,
  title={Optimal Gradient Checkpoint Search for Arbitrary Computation Graphs},
  author={Jianwei Feng and Dong Huang},
  booktitle={CVPR},
  year={2021}
}
```

**Regular Training vs. Gradient CheckPointing(GCP) Training:** **(a)** The regular training stores all tensors during forward, and uses these tensors to compute gradients during backward. **(b)** GCP stores a subset of tensors during the first forward, and conducts extra local re-forwards to compute tensors and gradients during backward. Our approach automatically searches the optimal set of Gradient Checkpoints (GCs) for memory cut-off. Such that on the same **physical** GPU memory (e.g., in 4 RTX2080Ti GPUs), GCP training can accommodate models that require **2+** times extra GPU memory. 

![scheme_compare](./figures/scheme_compare_gradient_checkpoint.png)<!-- .element height="20%" width="20%" -->
![table_compare](./figures/table_compare_gradient_checkpoint.png)<!-- .element height="20%" width="20%" -->

# Reducing Training Memory by Optimal Gradient Checkpointing
| Model Name |   Input Size  |    Regular Training Memory (MB)    |     OGC Training Memory (MB)    |    Memory Cut off   |   Regular Training Time (s)  |    OGC Training Time (s)  |   Time Overhead   |
|:------------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
|       Alexnet      |      (1024, 3, 224, 224)      |      4955      |      3287      |      34%      |      0.388      |      0.519      |       34%      |
|      VGG 11      |     (64, 3, 224, 224)     |     3577     |     2781     |     22%    |     0.266     |     0.356     |      34%      |
|     VGG 13    |  (64, 3, 224, 224) | 5136 | 3565 | 31% | 0.418 | 0.558 |      33%      |
|     VGG 16     |   (64, 3, 224, 224)   |   5136   |   3565   |   31%   |   0.503   |   0.666   |      32%      |
|    VGG 19    |      (64, 3, 224, 224)      |      5189      |    3565   |       31%      |    0.581   |    0.774   |      33%      |
|     Resnet 18    |    (256, 3, 224, 224)    |    5635    |    3677    |    35%    |    0.422    |    0.548    |      30%      |
|     Resnet 34    |  (128, 3, 224, 224)  |   4079  |  1838 |  55%  |  0.364 |  0.493  |      35%      |
|    Resnet 50    |  (64, 3, 224, 224) |   5323   |  1973 |  63%  |  0.394 |  0.516  |      31%      |
|    Resnet 101   |  (32, 3, 224, 224)  |  3934  |  1024  |  74%  |  0.356  |  0.482  |      35%      |
|    Resnet 152       |   (16, 3, 224, 224)  |     2767     |   526   |   81%    |   0.241  |   0.331  |      37%      |
|  Densenet 121  |     (32, 3, 224, 224)    |     4027     |    898    |     78%    |     0.218    |     0.292    |      34%      |
|  Densenet 161  |    (16, 3, 224, 224)    |    3751   |     666    |   82%   |    0.252    |   0.341   |       36%      |
|    Densenet 169       |     (32, 3, 224, 224)   |    4862   |    897   |    82%    |    0.270   |    0.357   |      32%      |
|    Densenet 201    | (16, 3, 224, 224) |  3146  |  474 |  85% | 0.200 | 0.306 |      53%      |
|    Inception v3    | (32, 3, 300, 300)|  3074  |  881 |  71% | 0.291 | 0.374 |      29%      |
|    NASNet Cifar10   | (64, 3, 32, 32) |  5832  |  1129 |  81% | 0.408 | 0.535 |      31%      |
|    AmoebaNet Cifar10    | (64, 3, 32, 32) |  4944  |  1058 |  79% | 0.331 | 0.450 |      36%      |
|    DARTS Cifar10    | (64, 3, 32, 32) |  5627  |  1115 |  80% | 0.318 | 0.494 |      55%      |

The memory numbers in the table are calculated after removing stationary memory cost such as model weights and pytorch cuda interface.

## Updates
Next step: More tests and bug fix in automatic computation graph parser

2021.06.17, we implemented automatic computation graph parser using torch.jit! Now user can input an arbitrary pytorch model (nn.module) 
and runs optimal gradient checkpointing solver without manually parsing the computation graph.

## Installation

### Requirements:

- Python > 3.x
- Pytorch >= 1.5

### Step-by-step

Install pytorch 1.5 from https://pytorch.org/

Install other dependencies.
```
pip install -r requirements.txt
```

## Usage
### Benchmark Time and Memory Performance of Optimal Gradient Checkpointing
Run Optimal Gradient Checkpointing on Resnet101 and benchmark training memory cost and time cost
```
python benchmark.py --arch resnet101 --device cuda:0
```

### Train Model with Optimal Gradient Checkpointing
An example of training on CIFAR 10 dataset can be seen at [train_cifar10.ipynb](train_cifar10.ipynb)

The script will do the following:
1. Define a pytorch model (nn.Module) and its inputs
1. Parse the computation graph of the pytorch model with torch.jit, runs optimal gradient checkpointing algorithm, and returns a new 
model whose forward function will execute gradient checkpointing training with optimal gradient checkpoints
1. Run gradient checkpointing training and evaluation on CIFAR 10 for 2 epochs

When defining the network, we highly recommend using modules and operations in torch.nn instead of in torch.nn.functional, 
for example use torch.nn.ReLU instead of torch.nn.functional.relu. 

So far our computation graph parser can handle operations in torch.nn pretty well but might be buggy with torch.nn.functional.
We will continue to improve it.

### Implement parse_graph function for Custom Network
You can also implement your own parse_graph function to create computation graph for your network.

An example of custom model and parse_graph function can be seen at [manual_parse_graph.ipynb](manual_parse_graph.ipynb)


