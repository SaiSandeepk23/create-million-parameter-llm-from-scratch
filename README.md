# 2.3MParams-LLM-From-Scratch-Python

<a href="https://colab.research.google.com/drive/1AlnGsNU3BauFhn3ZY6tk71G9ANUAQRtx?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

<img src="https://i.ibb.co/r56NHtM/1-ox3h-To-PFUWx-Aw-URx-YEXi-Gg-removebg-preview.png" alt="LLM Architecture Diagram">

Creating your own Large Language Model (LLM) is a significant undertaking that many major technology companies are currently pursuing. While models often range from 7 billion to 70 billion parameters, there is immense value in understanding the underlying mechanics through smaller-scale implementations. Many resources focus heavily on theory without providing the practical, executable code necessary for development.

This project provides a detailed implementation and expansion of the LLaMA-from-scratch architecture originally proposed by Brian Kitano. It serves as a comprehensive guide for building and training a 2.3 million parameter model.

In this repository, we develop an LLM with 2.3 million parameters that can be trained without the need for high-end GPU hardware. Following the LLaMA 1 Paper approach, we use a simplified dataset to demonstrate the accessibility of creating million-parameter models.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Understanding the Transformer Architecture of LLaMA](#understanding-the-transformer-architecture-of-llama)
  - [Pre-normalization Using RMSNorm](#pre-normalization-using-rmsnorm)
  - [SwiGLU Activation Function](#swiglu-activation-function)
  - [Rotary Embeddings (RoPE)](#rotary-embeddings-rope)
- [Setting the Stage](#setting-the-stage)
- [Data Preprocessing](#data-preprocessing)
- [Evaluation Strategy](#evaluation-strategy)
- [Setting Up a Base Neural Network Model](#setting-up-a-base-neural-network-model)
- [Replicating LLaMA Architecture](#replicating-llama-architecture)
  - [RMSNorm for pre-normalization](#rmsnorm-for-pre-normalization)
  - [Rotary Embeddings](#rotary-embeddings)
  - [SwiGLU activation function](#swiglu-activation-function)
- [Experimenting with hyperparameters](#experimenting-with-hyperparameters)
- [Saving Your Language Model (LLM)](#saving-your-language-model-llm)
- [Conclusion](#conclusion)

## Prerequisites

A basic understanding of object-oriented programming (OOP) and neural networks (NN) is required. Familiarity with PyTorch is essential for following the implementation.

| Topic               | Video Link                                                |
|---------------------|-----------------------------------------------------------|
| OOP                 | [OOP Video](https://www.youtube.com/watch?v=Ej_02ICOIgs) |
| Neural Network      | [Neural Network Video](https://www.youtube.com/watch?v=Jy4wM2X21u0) |
| Pytorch             | [Pytorch Video](https://www.youtube.com/watch?v=V_xro1bcAuA) |

## Understanding the Transformer Architecture of LLaMA

To build an LLM using the LLaMA approach, we must first understand its architectural deviations from the vanilla transformer.

<img src="https://cdn-images-1.medium.com/max/25620/1*nt-ydHhSVsaLXq_HZRaLQA.png" alt="Difference between Transformers and Llama architecture" style="width: 50%;">

For those unfamiliar with the standard transformer architecture, you can refer to foundational guides on attention mechanisms and encoder-decoder structures.

Key LLaMA concepts include:

### Pre-normalization Using RMSNorm:

LLaMA employs RMSNorm for normalizing the input of each transformer sub-layer. Inspired by GPT-3, this method optimizes computational costs. RMSNorm provides similar performance to LayerNorm but reduces running time by approximately 7% to 64%.

<img src="https://cdn-images-1.medium.com/max/3604/1*9FA6P93WhRuWFXxVlPG3LA.png" alt="Root Mean Square Layer Normalization" style="width: 50%;">

It simplifies LayerNorm by removing the mean statistic and focusing on re-scaling invariance based on the root mean square (RMS) statistic.

### SwiGLU Activation Function:

LLaMA introduces the SwiGLU activation function, which extends the Swish activation function. It involves a custom layer with a dense network to split and multiply input activations, enhancing the expressive power of the model.

<img src="https://cdn-images-1.medium.com/max/13536/1*N3dwnqNUD0TdwPYO0NlhYg.png" alt="SwiGLU Variants" style="width: 50%;">

### Rotary Embeddings (RoPE):

Rotary Embeddings encode absolute positional information using a rotation matrix. This allows for explicit relative position dependency in self-attention. RoPE offers scalability to various sequence lengths and naturally decaying inter-token dependency as distances increase.

The LLaMA paper also utilizes the AdamW optimizer, causal multi-head attention operators, and optimized backward functions to improve training efficiency.

## Setting the Stage

We utilize standard Python libraries for this implementation:

```python
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
import urllib.request
```

The model parameters are managed via a configuration object:

```python
MASTER_CONFIG = {
    # Parameters defined during implementation
}
```

## Data Preprocessing

While the original LLaMA was trained on 1.4 trillion tokens, we use the TinyShakespeare dataset (approximately 1 million characters) for this scaled-down version.

```python
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_name = "tinyshakespeare.txt"
urllib.request.urlretrieve(url, file_name)
```

We establish character-level tokenization:

```python
lines = open("tinyshakespeare.txt", "r").read()
vocab = sorted(list(set(lines)))
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

def encode(s):
    return [stoi[ch] for ch in s]

def decode(l):
    return "".join([itos[i] for i in l])
```

The dataset is converted to a PyTorch tensor and split into training, validation, and test sets using a sliding context window.

```python
MASTER_CONFIG.update({
    "batch_size": 8,
    "context_window": 16
})
```

## Evaluation Strategy

We implement a loss-based evaluation function to monitor performance across training and validation splits without computing gradients.

```python
@torch.no_grad()
def evaluate_loss(model, config=MASTER_CONFIG):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(dataset, split, config["batch_size"], config["context_window"])
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out
```

## Setting Up a Base Neural Network Model

We begin with a basic model and iteratively apply LLaMA-specific enhancements.

```python
class SimpleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.linear = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]),
            nn.ReLU(),
            nn.Linear(config["d_model"], config["vocab_size"]),
        )

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        logits = self.linear(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config["vocab_size"]), targets.view(-1))
            return logits, loss
        return logits
```

## Replicating LLaMA Architecture

We integrate the three primary modifications: RMSNorm, Rotary Embeddings, and SwiGLU.

### RMSNorm for pre-normalization:

```python
class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw
```

### Rotary Embeddings:

We implement the rotation matrix to encode positional data within the attention heads.

```python
def get_rotary_matrix(context_window, embedding_dim):
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    for position in range(context_window):
        for i in range(embedding_dim // 2):
            theta = 10000. ** (-2. * (i - 1) / embedding_dim)
            m_theta = position * theta
            R[position, 2 * i, 2 * i] = np.cos(m_theta)
            R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
    return R
```

### SwiGLU activation function:

Replacing ReLU with SwiGLU improves the model's ability to capture complex patterns.

```python
class SwiGLU(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        return swish_gate * self.linear(x)
```

## Final Llama Model

The final architecture combines these elements into a multi-layer LlamaBlock structure.

```python
class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config["vocab_size"], config["d_model"])
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config["n_layers"])])
        )
        self.ffn = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]),
            SwiGLU(config["d_model"]),
            nn.Linear(config["d_model"], config["vocab_size"]),
        )
```

## Experimenting with hyperparameters

We can optimize training using learning rate schedulers like Cosine Annealing.

```python
llama_optimizer = torch.optim.Adam(
    llama.parameters(),
    betas=(.9, .95),
    weight_decay=.1,
    eps=1e-9,
    lr=1e-3
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(llama_optimizer, 300, eta_min=1e-5)
```

## Saving Your Language Model (LLM)

Models can be saved for future inference or converted for use with the Hugging Face Transformers library.

```python
# Save model parameters
torch.save(llama.state_dict(), "llama_model_params.pth")

# Save for Transformers library
llama_transformers.save_pretrained("llama_model_transformers")
```

## Conclusion

This project demonstrates the step-by-step implementation of the LLaMA architecture. By scaling the model to 2.3 million parameters, we achieve a balance between computational efficiency and linguistic capability. For further improvements, increasing the parameter count to the 10M-20M range typically yields significantly better language comprehension.

## Maintainer

**Sai Sandeep Kethiboina**
AI and Machine Learning Engineer

AI/ML Engineer and Data Scientist with over 5 years of experience designing, developing, and deploying scalable Artificial Intelligence, Machine Learning, and Generative AI solutions. Specialized in building data-driven products and predictive systems using Python, PyTorch, and TensorFlow across Telecommunications, Banking, and Healthcare domains.