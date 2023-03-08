import torch.nn as nn

class FeedFoward(nn.Module):
    def __init__(
        self,
        dim: int, 
        inner_dim_scale: int,
        dropout: float = .0) -> None:
        super().__init__()

        self.linear_seq = nn.Sequential(
            nn.Linear(dim, inner_dim_scale),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim_scale, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.linear_seq(x)