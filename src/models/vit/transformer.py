import torch.nn as nn

from attention import Attention
from feedfoward import FeedFoward
from normalization import LayerN

class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_inner_dim: int,
        dropout: float = .0
        ) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    LayerN(dim, Attention(dim, heads, dim_head, dropout)),
                    LayerN(dim, FeedFoward(dim, mlp_inner_dim, dropout))
                ])
            )

    def forward(self, x):
        for attn, feedf in self.layers:
            x = attn(x) + x
            x = feedf(x) + x
            return x
    