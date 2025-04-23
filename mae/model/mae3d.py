import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=4, in_channels=1, embed_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # 输入 x: [B, 1, D, H, W]
        x = self.proj(x)  # 输出: [B, embed_dim, D', H', W']
        x = rearrange(x, 'b c d h w -> b (d h w) c')  # 输出: [B, N, embed_dim]
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        # 输入 x: [B, N, dim]
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class MAE3D(nn.Module):
    def __init__(self, img_size=16, patch_size=4, in_channels=1, embed_dim=512, depth=8):  # 增加 Transformer 深度
        super().__init__()
        self.patch_embed = PatchEmbed3D(patch_size, in_channels, embed_dim)
        self.encoder_blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim) for _ in range(depth)
        ])  # 增加 Transformer 层数
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # 解码器：确保输出尺寸与输入图像匹配
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1),  # 还原尺寸
            nn.ReLU(),
            nn.ConvTranspose3d(embed_dim // 2, 1, kernel_size=4, stride=2, padding=1),  # 进一步还原
        )
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_num = (img_size // patch_size) ** 3

    def forward(self, x):
        patches = self.patch_embed(x)           # [B, N, embed_dim]
        encoded = self.encoder_blocks(patches)  # [B, N, embed_dim]
        encoded = self.encoder_norm(encoded)    # [B, N, embed_dim]

        # 解码器处理：将编码的特征从 [B, N, embed_dim] 转换为 [B, 1, D', H', W']
        B, N, embed_dim = encoded.shape
        patch_size = self.patch_size

        # 这里将 N 恢复成 立方体形式 (D, H, W)
        grid_size = round(N ** (1/3))  # 计算网格尺寸
        assert grid_size ** 3 == N, f"Expected cubic number of patches, got N={N}"
        recon = encoded.view(B, grid_size, grid_size, grid_size, embed_dim)
        recon = recon.permute(0, 4, 1, 2, 3).contiguous()  # [B, embed_dim, D', H', W']

        # 使用 ConvTranspose3d 进行上采样恢复尺寸
        recon = self.decoder(recon)  # [B, 1, D, H, W]

        loss = nn.MSELoss()(recon, x)  # 与输入图像进行 MSE 比较
        return loss, recon

    def encode_only(self, x):
        patches = self.patch_embed(x)           # [B, N, embed_dim]
        encoded = self.encoder_blocks(patches)
        encoded = self.encoder_norm(encoded)
        return encoded  # 返回未经过池化的编码特征 [B, N, embed_dim]
