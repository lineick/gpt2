import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt_attn import CausalSelfAttention
from gpt_config import GPTConfig
from gpt_mlp import MLP


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # in gpt2, layernorm before the attention blocks, not after (reason clean res stream from input to output)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
