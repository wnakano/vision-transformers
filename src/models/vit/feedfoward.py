import torch.nn as nn
from typing import Literal, Union

class FeedFoward(nn.Module):
    def __init__(
        self,
        dim: int, 
        inner_dim_scale: int,
        dropout: float = .0,
        act_fn: Literal["GELU", "ReLU", "SELU"] = "GELU") -> None:
        super().__init__()

        _act_fn = getattr(nn, act_fn)

        self.linear_seq = nn.Sequential(
            nn.Linear(dim, inner_dim_scale),
            _act_fn(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim_scale, dim),
            nn.Dropout(dropout)
        )
    


    def forward(self, x):
        return self.linear_seq(x)