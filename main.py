import pathlib
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

# from torchvision import transforms
from torchvision import datasets, transforms, utils
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from kan import KAN
from collections import OrderedDict


class Config:
    lr = 0.0002
    max_epoch = 8
    batch_size = 32
    z_dim = 100
    image_size = 64
    g_conv_dim = 64
    d_conv_dim = 64
    log_step = 100
    sample_step = 500
    sample_num = 32
    IMAGE_PATH = "../dataset/img_align_celeba/img_align_celeba"
    SAMPLE_PATH = "./samples"
    KAN_MODEL_IMAGE_PATH = "./figures"


class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, conv_dim, 4)
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
    def __init__(self, device: torch.device, z_dim=256, image_size=128, conv_dim=64):
        super().__init__()
        # KAN based generator
        self.generator = KAN(
            width=[image_size**2, conv_dim * 4, conv_dim],
            k=3,
            seed=0,
            grid=5,
            device=device,
        )
        # Discriminator
        self.discriminator = Discriminator(image_size, conv_dim)

    def forward(self, x):
        return self.discriminator(x)

    def loss_function(self, y_hat, y):
        return nn.BCELoss()(y_hat, y)

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Scale(Config.image_size),
                # transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dataset = ImageFolder(Config.IMAGE_PATH, transform)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )
        return data_loader

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=Config.lr, betas=(0.4, 0.999)
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=Config.lr, betas=(0.4, 0.999)
        )

        # return the list of optimizers and second empty list is for schedulers (if any)
        return [optimizer_G, optimizer_D], []

        # Training Loop

    def training_step(self, batch, batch_idx, optimizer_idx):
        # batch returns x and y tensors
        real_images, _ = batch

        # ground truth (tensors of ones and zeros) same shape as images
        valid = torch.ones(real_images.size(0), 1)
        fake = torch.zeros(real_images.size(0), 1)

        # svaing loss_function as local variable
        criterion = self.loss_function

        # As there are 2 optimizers we have to train for both using 'optimizer_idx'
        ## Generator
        if optimizer_idx == 0:
            # Generating Noise (input for the generator)
            gen_input = torch.randn(real_images.shape[0], 100)

            # Converting noise to images
            self.gen_images = self.generator(gen_input)

            # Calculating generator loss
            # How well the generator can create real images
            g_loss = criterion(self(self.gen_images), valid)

            # for output and logging purposes (return as dictionaries)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict(
                {
                    "loss": g_loss,
                    "progress_bar": tqdm_dict,
                    "log": tqdm_dict,
                    "g_loss": g_loss,
                }
            )
            return output
        ## Discriminator
        if optimizer_idx == 1:
            # Calculating disciminator loss
            # How well discriminator identifies the real and fake images
            real_loss = criterion(self(real_images), valid)
            fake_loss = criterion(self(self.gen_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2.0

            # for output and logging purposes (return as dictionaries)
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict(
                {
                    "loss": d_loss,
                    "progress_bar": tqdm_dict,
                    "log": tqdm_dict,
                    "d_loss": d_loss,
                }
            )
            return output

        # calls after every epoch ends

    def on_epoch_end(self):
        # Saving 5x5 grid
        utils.save_image(
            self.gen_images.data[:25],
            Config.SAMPLE_PATH + "/%d.png" % self.current_epoch,
            nrow=5,
            padding=0,
            normalize=True,
        )


if __name__ == "__main__":
    if not os.path.exists(Config.SAMPLE_PATH):
        os.makedirs(Config.SAMPLE_PATH)
    if not os.path.exists(Config.KAN_MODEL_IMAGE_PATH):
        os.makedirs(Config.KAN_MODEL_IMAGE_PATH)
    gan = KalmagorovArnoldGAN(
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    trainer = pl.Trainer(max_epochs=Config.max_epoch, fast_dev_run=False)
    trainer.fit(gan)
