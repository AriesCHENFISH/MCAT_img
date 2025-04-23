import torch
import torch.nn as nn
from einops import rearrange

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, patch_size, img_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Embedding(self.num_patches, embed_dim)

    def forward(self):
        return self.pos_embedding.weight  # [N, embed_dim]

# 2D Patch 嵌入
class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_channels=1, embed_dim=512, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_encoding = PositionalEncoding(embed_dim, patch_size, img_size)
        
        # 新增 Linear 层，将拼接后的维度映射回 embed_dim
        self.fc = nn.Linear(embed_dim + 4, embed_dim)  # embed_dim + 4 来自拼接 ROI 信息

    def forward(self, x, roi_embedding=None):
        x = self.proj(x)  # [B, C, H/P, W/P]
        x = rearrange(x, 'b c h w -> b (h w) c')  # 展开为 [B, N, embed_dim]
        
        # 添加位置编码
        pos_encoding = self.pos_encoding()  # 获取位置编码
        x = x + pos_encoding  # 加入位置编码
        
        # 如果有 ROI 信息，则将 ROI 嵌入加入到特征中
        if roi_embedding is not None:
            roi_embedding_expanded = roi_embedding.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, N, 4]
            x = torch.cat((x, roi_embedding_expanded), dim=-1)  # 拼接 ROI 信息

            # 使用 Linear 层将拼接后的维度映射回 embed_dim
            x = self.fc(x)  # [B, N, embed_dim]
        return x

# Transformer 编码器块
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

# MAE2D 模型
class MAE2D(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=1, embed_dim=512, depth=4):
        super().__init__()
        self.patch_embed = PatchEmbed2D(patch_size, in_channels, embed_dim, img_size)
        self.encoder = nn.Sequential(*[TransformerBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, patch_size**2),
        )
        self.patch_size = patch_size
        self.img_size = img_size

    def forward(self, x, roi_embedding=None):
        patches = self.patch_embed(x, roi_embedding)  # 添加 ROI 嵌入
        encoded = self.encoder(patches)
        encoded = self.norm(encoded)
        recon_patches = self.decoder(encoded)  # [B, N, P^2]
        recon = recon_patches.view(x.shape[0], 1, self.img_size, self.img_size)
        loss = nn.MSELoss()(recon, x)
        return loss, recon

    def encode_only(self, x, roi_embedding=None):
        patches = self.patch_embed(x, roi_embedding)  # 添加 ROI 嵌入
        encoded = self.encoder(patches)
        encoded = self.norm(encoded)
        return encoded.mean(dim=1)
