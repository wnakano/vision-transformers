
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from einops import repeat
from typing import Tuple

class Embeddings(nn.Module):
    def __init__(
        self,
        num_patches: int,
        linear_projection_dim: int,
        patch_size: Tuple[int, int],
        num_channels: int = 3,
        emb_dropout: float = .0
        ) -> None:
        super().__init__()
        patch_H, patch_W = patch_size 

        self.to_patch_embeddings = nn.Sequential(
            nn.Conv2d(
                num_channels, 
                linear_projection_dim, 
                kernel_size=(patch_H, patch_W),
                stride=(patch_H, patch_W)
            ),
            Rearrange('b k h w -> b (h w) k'),
            nn.LayerNorm(linear_projection_dim)
            # [ ] Here could be added a Linear layer to increase attention dim
        )

        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, linear_projection_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, linear_projection_dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        # patch embeddings
        embeddings = self.to_patch_embeddings(x)
        b, _, _ = embeddings.shape
        cls_token = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        # patch + class embeddings
        embeddings = torch.cat([embeddings, cls_token], dim=1)
        # combined (patch, class, positional) embeddings
        embeddings += self.positional_embedding
        embeddings = self.dropout(embeddings)
        return embeddings