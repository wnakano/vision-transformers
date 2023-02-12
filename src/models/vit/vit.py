
import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Rearrange
from typing import Tuple

from transformer import Transformer

class ViT(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = "cls",
        num_channels: int = 3,
        dim_head: int = 64,
        dropout: float = .0,
        emb_dropout: float = .0
        ) -> None:
        super().__init__()

        im_H, im_W = image_size
        patch_H, patch_W = patch_size
        
        assert im_H % patch_H == 0 and im_W % patch_W == 0, "."

        num_patches = (im_H // patch_H) * (im_W // patch_W)
        linear_projection_dim = patch_H * patch_W * num_channels
        
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

        self.to_latent = nn.Identity()
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, linear_projection_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, linear_projection_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=linear_projection_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_inner_dim=mlp_dim,
            dropout=dropout
        )

        self.pool = pool
        self.head = nn.Sequential(
            nn.LayerNorm(linear_projection_dim),
            nn.Linear(linear_projection_dim, num_classes)
        )

    def forward(self, x):
        patch_embeddings = self.to_patch_embeddings(x)
        b, num_lin_proj, dim_lin_proj_patch = patch_embeddings.shape
        # num_lin_proj is only used if patch_embeddings has dim=1 > linear_projection_dim

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        patch_embeddings_cls = torch.cat((cls_tokens, patch_embeddings), dim=1)
        combined_embeddings = patch_embeddings_cls + self.positional_embedding
        
        x = self.dropout(combined_embeddings)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        x = self.head(x)
        
        return x


if __name__ == "__main__":
    x = torch.randn(4, 3, 224, 224)
    vit = ViT(
        image_size=(224, 224),
        patch_size=(16, 16),
        num_classes=5,
        num_channels=3,
        depth=4,
        dim=10,
        heads=12,
        mlp_dim=100
    )

    y = vit(x)