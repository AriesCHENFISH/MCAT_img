import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=16, in_channels=1, embed_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H/P, W/P]
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class MAE2D(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=512, depth=4):
        super().__init__()
        self.patch_embed = PatchEmbed2D(patch_size, in_channels, embed_dim)
        self.encoder = nn.Sequential(*[TransformerBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, patch_size**2),
        )
        self.patch_size = patch_size
        self.img_size = img_size

    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x)
        encoded = self.encoder(patches)
        encoded = self.norm(encoded)
        recon_patches = self.decoder(encoded)  # [B, N, P^2]
        recon = recon_patches.view(B, 1, self.img_size, self.img_size)
        loss = nn.MSELoss()(recon, x)
        return loss, recon

    def encode_only(self, x):
        patches = self.patch_embed(x)
        encoded = self.encoder(patches)
        encoded = self.norm(encoded)
        return encoded.mean(dim=1)
