import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, conv_dim, 4, bn=False)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim * 2, 4)
        self.conv3 = nn.Conv2d(conv_dim * 2, conv_dim * 4, 4)
        self.conv4 = nn.Conv2d(conv_dim * 4, conv_dim * 8, 4)
        self.fc = nn.Conv2d(conv_dim * 8, 1, int(image_size / 16), 1, 0, False)

    def forward(self, x):  # if image_size is 64, output shape is below
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 32, 32)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 512, 4, 4)
        out = F.sigmoid(self.fc(out)).squeeze()
        return out


class KalmagorovArnoldGAN(pl.LightningModule):
    def __init__(self):
        pass
