
import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Reduce
from typing import Tuple

from embeddings import Embeddings
from transformer import EncoderTransformer
from cls_head import ClsHead

class ViT(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        num_classes: int,
        depth: int,
        heads: int,
        inner_dim_scale: int,
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

        self.embeddings = Embeddings(
            num_patches=num_patches, 
            linear_projection_dim=linear_projection_dim, 
            patch_size=patch_size,
            num_channels=num_channels,
            emb_dropout=emb_dropout
        )

        self.transformer = EncoderTransformer(
            dim=linear_projection_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            inner_dim_scale=inner_dim_scale,
            dropout=dropout
        )

        self.head = ClsHead(
            linear_projection_dim=linear_projection_dim, 
            num_classes=num_classes
        )

    def forward(self, x):
        x = self.embeddings(x)
        x = self.transformer(x)
        x = self.head(x)
        
        return x


if __name__ == "__main__":
    x = torch.randn(4, 3, 224, 224)
    vit = ViT(
        image_size=(224, 224),
        patch_size=(16, 16),
        inner_dim_scale=4,
        num_classes=5,
        num_channels=3,
        depth=10,
        heads=12
    )
    y = vit(x)