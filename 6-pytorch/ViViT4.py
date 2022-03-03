import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange


class FDAttention(nn.Module):
    def __init__(self, nt, nh, nw, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        project_out = not (heads == 1 and dim_head == dim)

        self.nt = nt
        self.nh = nh
        self.nw = nw
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.layer_norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.W_qkv = nn.Linear(dim, heads * dim_head  * 3, bias=False)
        self.W_o = nn.Sequential(
            nn.Linear(heads * dim_head, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):   # x: [B, nt * nh * nw, dim]
        b = x.shape[0]
        h = self.heads
        residual = x

        x = self.layer_norm(x)
        qkv = self.W_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        qs, qt = q.chunk(2, dim=1)
        ks, kt = k.chunk(2, dim=1)
        vs, vt = v.chunk(2, dim=1)

        # Attention over spatial heads
        qs = qs.view(b, h // 2, self.nt, self.nh * self.nw, -1)
        ks = ks.view(b, h // 2, self.nt, self.nh * self.nw, -1)
        vs = vs.view(b, h // 2, self.nt, self.nh * self.nw, -1)
        spatial_dots = einsum('b h t i d, b h t j d -> b h t i j', qs, ks) * self.scale
        spatial_attn = self.softmax(spatial_dots)
        spatial_out = einsum('b h t i j, b h t j d -> b h t i d', spatial_attn, vs)
        spatial_out = rearrange(spatial_out, 'b h t hw d -> b (t hw) (h d)')

        # Attention over temporal heads
        qt = qt.view(b, h // 2, self.nh * self.nw, self.nt, -1)
        kt = kt.view(b, h // 2, self.nh * self.nw, self.nt, -1)
        vt = vt.view(b, h // 2, self.nh * self.nw, self.nt, -1)
        temporal_dots = einsum('b h s i d, b h s j d -> b h s i j', qt, kt) * self.scale
        temporal_attn = self.softmax(temporal_dots)
        temporal_out = einsum('b h s i j, b h s j d -> b h s i d', temporal_attn, vt)
        temporal_out = rearrange(temporal_out, 'b h hw t d -> b (t hw) (h d)')

        # concatenate
        out = torch.cat((spatial_out, temporal_out), dim=-1)    # [B, nt * nh * nw, dim_head * heads]
        out = self.W_o(out) + residual
        return out


class FDAEncoder(nn.Module):
    """ Model4: Factorized Dot-Product Attention """
    def __init__(self, nt, nh, nw, depth, dim, heads, dim_head, dropout=0.):
        super().__init__()
        self.nt = nt
        self.nh = nh
        self.nw = nw
        
        self.layers = nn.ModuleList([
            FDAttention(nt, nh, nw, dim, heads, dim_head, dropout) for _ in range(depth)
        ])

    def forward(self, x):
        for attn in self.layers:
            x = attn(x)
        return x

class ViViT(nn.Module):
    """Model-4 backbone of ViViT """
    def __init__(self, T, H, W, t, h, w, num_classes=10, 
                depth=1, dim=512, heads=8, dim_head=64,
                mode='tubelet', emb_dropout=0., dropout=0.):
        super().__init__()
        assert T % t == 0 and H % h == 0 and W % w == 0, "Video dimensions should be divisible by tubelet size "
        assert heads % 2 == 0, "Number of head should be even"
        nt, nh, nw = T // t, H // h, W // w   

        self.tubelet_embedding = nn.Sequential(
            Rearrange('b c (nt t) (nh h) (nw w) -> b (nt nh nw) (t h w c)', t=t, h=h, w=w),
            nn.Linear(t * h * w * 3, dim)
        )
        # repeat same spatial position encoding temporally
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, nh * nw, dim)).repeat(1, nt, 1, 1)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = FDAEncoder(nt, nh, nw, depth, dim, heads, dim_head, dropout)

        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        """ x is a video: (B, C, T, H, W) """
        x = self.tubelet_embedding(x)
        # tokens += self.pos_embedding
        x = self.dropout(x)   # [B, nt * nh * nw, d]

        out = self.transformer(x)
        out = out.mean(dim=1)

        out = self.to_latent(out)
        out = self.mlp_head(out)
        return out

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(1, 3, 32, 64, 64)

    model = ViViT(
        T=32, H=64, W=64,
        t=8, h=4, w=4,
    )
    out = model(x)  
    print(out.shape)
