import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(self.layer_norm(x)) + x

class FSAttention(nn.Module):
    """ Model3: Factorized Self-Attention """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.layer_norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.W_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.W_o = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        residual = x

        x = self.layer_norm(x)
        qkv = self.W_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dot = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.softmax(dot)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.W_o(out) + residual
        return out


class FSAEncoder(nn.Module):
    def __init__(self, nt, nh, nw, depth, dim, heads, dim_head, dim_mlp, dropout=0.):
        super().__init__()
        self.nt = nt
        self.nh = nh
        self.nw = nw
        self.layers = nn.ModuleList(nn.ModuleList([
            FSAttention(dim, heads, dim_head, dropout),
            FSAttention(dim, heads, dim_head, dropout),
            FeedForward(dim, dim_mlp, dropout)]) for _ in range(depth))
            
    def forward(self, x):  # x: [B, nt, nh*nw, d]
        # print(self.layers)
        batch = x.shape[0]
        # extract spatial tokens from x
        x = torch.flatten(x, start_dim=0, end_dim=1)                        # [B*nt, nh*nw, d]

        for sp_attn, temp_attn, ff in self.layers:
            sp_attn_x = sp_attn(x)                                          # [B*nt, nh*nw, d]

            # Reshape tensors for temporal attention
            sp_attn_x = sp_attn_x.chunk(batch, dim=0)                       # B * [nt, nh*nw, d]
            sp_attn_x = [temp[None] for temp in sp_attn_x]                  # B * [1, nt, nh*nw, d]
            sp_attn_x = torch.cat(sp_attn_x, dim=0).transpose(1, 2)         # [B, nh*nw, nt, d]
            sp_attn_x = torch.flatten(sp_attn_x, start_dim=0, end_dim=1)    # [B*nh*nw, nt, d]

            temp_attn_x = temp_attn(sp_attn_x) 
            x = ff(temp_attn_x)

            # Reshape tensor again for spatial attention
            x = x.chunk(batch, dim=0)                                       # B * [nh*nw, nt, d]
            x = [temp[None] for temp in x]                                  # B * [1, nh*nw, nt, d]
            x = torch.cat(x, dim=0).transpose(1, 2)                         # [B, nt, nh*nw, d]
            x = torch.flatten(x, start_dim=0, end_dim=1)                    # [B*nt, nh*nw, d]

        # Finally, reshape to [B, nt*nh*nw, d]
        x = x.chunk(batch, dim=0)                                           # B * [nt, nh*nw, d]
        x = [temp[None] for temp in x]                                      # B * [1, nt, nh*nw, d]
        x = torch.cat(x, dim=0)                                             # [B, nh, nt*nw, d]
        x = torch.flatten(x, start_dim=1, end_dim=2)                        # [B, nt*nh*nw, d]
        return x

class ViViT(nn.Module):
    """Model-3 backbone of ViViT """
    def __init__(self, T, H, W, t, h, w, num_classes=10, 
                depth=1, heads=8, dim_head=64, dim=512, dim_mlp=2048,
                mode='tubelet', emb_dropout=0., dropout=0.):
        super().__init__()
        assert T % t == 0 and H % h == 0 and W % w == 0, "Video dimensions should be divisible by tubelet size "
        nt, nh, nw = T // t, H // h, W // w   

        self.tubelet_embedding = nn.Sequential(
            Rearrange('b c (nt t) (nh h) (nw w) -> b nt (nh nw) (t h w c)', t=t, h=h, w=w),
            nn.Linear(t * h * w * 3, dim)
        )
        # repeat same spatial position encoding temporally
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, nh * nw, dim)).repeat(1, nt, 1, 1)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = FSAEncoder(nt, nh, nw, depth, dim, heads, dim_head, dim_mlp, dropout)
                                                    
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        """ x is a video: (B, C, T, H, W) """
        x = self.tubelet_embedding(x)
        x += self.pos_embedding
        x = self.dropout(x)   # [B, nt, nh * nw, dim]
        print(x.shape)

        out = self.transformer(x)
        print(out.shape)
        out = out.mean(dim=1)
        print(out.shape)
        out = self.to_latent(out)
        out = self.mlp_head(out)
        return out

if __name__ == '__main__':
    x = torch.rand(1, 3, 32, 64, 64)

    model = ViViT(
        T=32, H=64, W=64,
        t=8, h=4, w=4,
    )
    out = model(x)  # (B, 10)