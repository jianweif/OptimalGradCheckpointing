# Optimal Gradient Checkpoint Search
This is the official implementation of the paper: Jianwei Feng and Dong Huang, Optimal Gradient Checkpoint Search for Arbitrary Computation Graphs. [arXiv version](https://arxiv.org/abs/1808.00079)

Regular Training vs. Gradient CheckPointing(GCP) Training. (a) The regular training stores all tensors during forward, and uses these tensors to compute gradients during backward. (b) GCP stores a subset of tensors during the first forward, and conducts extra local re-forwards to compute tensors and gradients during backward. Our approach automatically searches the optimal set of Gradient Checkpoints (GCs) for memory cut-off. Such that on the same physical GPU memory (e.g., in 4 RTX2080Ti GPUs), GCP training can accommodate models that require 2+ times extra GPU memory. 

![scheme_compare](./figures/scheme_compare_gradient_checkpoint.png = 200x)
![table_compare](./figures/table_compare_gradient_checkpoint.png = 200x)

### Citation: 

```bash
@inproceedings{fenghuang2020,
  title={Optimal Gradient Checkpoint Search for Arbitrary Computation Graphs},
  author={Jianwei Feng and Dong Huang},
  booktitle={arXiv:1808.00079},
  year={2020}
}
```

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

## Run Optimal Gradient Checkpointing
```
python main.py
```
