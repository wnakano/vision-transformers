import torch.nn as nn
from typing import Literal

from self_attention import SelfAttention
from feedfoward import FeedFoward
from normalization import LayerN

class EncoderTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        inner_dim_scale: int,
        dropout: float = .0,
        act_fn: Literal["GELU", "ReLU", "SELU"] = "GELU"
        ) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    LayerN(dim, SelfAttention(dim, heads, dim_head, dropout)),
                    LayerN(dim, FeedFoward(dim, inner_dim_scale, dropout, act_fn))
                ])
            )

    def forward(self, x):
        for attn, feedf in self.layers:
            x = attn(x) + x
            x = feedf(x) + x
        return x
    