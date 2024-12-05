import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt_block import Block
from gpt_config import GPTConfig


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embed
                wpe=nn.Embedding(config.block_size, config.n_embd),  # positional embed
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )  # no bias as in gpt2 implementation
