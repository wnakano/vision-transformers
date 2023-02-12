import torch.nn as nn

class LayerN(nn.Module):
    def __init__(self, dim: int, branch: nn.Module) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.branch = branch

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.branch(x)
        return x