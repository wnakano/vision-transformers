
import torch
import torch.nn as nn

from einops.layers.torch import Reduce

class ClsHead(nn.Module):
    def __init__(
        self,
        linear_projection_dim: int,
        num_classes: int
        ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            Reduce("b n d -> b d", reduction="mean"),
            nn.LayerNorm(linear_projection_dim),
            nn.Linear(linear_projection_dim, num_classes)
        )

    def forward(self, x):
        x = self.head(x)
        return x