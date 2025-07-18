import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改为灰度图
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # 去掉 avgpool 和 fc

    def forward(self, x):
        return self.encoder(x)  # [B, 512, H/32, W/32]

class Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, 4, stride=2, padding=1),  # -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),           # -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),            # -> 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),             # -> 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),  # -> 224x224
            nn.Sigmoid(),  # normalize output to [0,1]
        )

    def forward(self, x):
        return self.decoder(x)

class ResNetAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        loss = nn.MSELoss()(recon, x)
        return loss, recon

    def encode_only(self, x):
        latent = self.encoder(x)
        return latent.mean(dim=[2,3])  # [B, C]
