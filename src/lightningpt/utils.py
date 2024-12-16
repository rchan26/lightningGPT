import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ModelArgs:
    n_embd: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_heads: int | None = None
    ffn_dim_multiplier: int = 4
    ffn_dropout: float = 0.0
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048
    vocab_size: int = -1
