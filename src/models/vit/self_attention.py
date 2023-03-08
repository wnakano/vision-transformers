import torch
import torch.nn as nn

from einops import rearrange
"""
└──
├──
"""
class SelfAttention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        heads: int = 8, 
        dim_head: int = 64,
        dropout: float = .0,
        expnd_inner_dim: bool = True) -> None:

        super().__init__()
        self._dim = dim
        self._heads = heads
        self._dim_head = dim_head

        _inner_dim = dim_head * heads if expnd_inner_dim else dim
        self._scale = dim_head ** -0.5
        self._to_qkv = nn.Linear(dim, _inner_dim * 3)
        self._attent = nn.Softmax(dim=-1)
        self._dropout = nn.Dropout(p=dropout)
        self._out = nn.Linear(_inner_dim, dim)

    def forward(self, x):
        # x = (b, (H // patch_size) ** 2 + 1, patch_size**2 * chn_in)
        #      _  ____________________  ____________________________
        #      |              |                      |       
        #      |              |                      └── linear projection dim for each patch: patch_size * patch_size * 3
        #      |              └── number of linearized patches + 1 (class token) 
        #      └── batch size
        # if patch_size = 16, H=W=224 > x = (b, (224//16)**2 + 1, 16**2 * 3)
        #                                   (b, 197, 768)

        qkv = self._to_qkv(x).chunk(3, dim=-1) # 3 tuples of (b, 197, linear_projection_dim / 3) 
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self._heads), qkv) # 3 tensor of (b, heads, num_lin_patches+1, dim_head)

        att_mmul = torch.matmul(q, k.transpose(-1, -2)) * self._scale # (b, heads, num_lin_patches+1, dim_head) * (b, heads, dim_head, num_lin_patches+1) -> (b, heads, num_lin_patches+1, num_lin_patches+1)  
        att = self._attent(att_mmul)
        att = self._dropout(att)

        out = torch.matmul(att, v) # (b, heads, num_lin_patches+1, dim_head) * (b, heads, num_lin_patches+1, num_lin_patches+1) > (b, heads, num_lin_patches+1, dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)') # (b, num_lin_patches+1, dim_heads * heads)
        
        out = self._out(out)
        out = self._dropout(out)

        return out

