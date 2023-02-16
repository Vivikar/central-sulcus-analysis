
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics

from monai.networks.nets import UNet, BasicUNet
from monai.losses import DiceLoss

import pytorch_lightning as pl
import torch

def get_criterions(name: str):
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif name == 'dice':
        return losses.DiceLoss()
    elif name == 'focal':
        return losses.FocalLoss(0.5)
    elif name == 'tversky':
        return losses.TverskyLoss(0.4, 0.4)
    else:
        raise ValueError(f'Unknown loss name: {name}')


class UNet3D(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.hparams = hparams

        self.unet = BasicUNet(spatial_dims=3,
                              in_channels=1,
                              out_channels=2,
                              features=(32, 32, 64, 128, 256, 32),
                              )

        self.criterion = DiceLoss(include_background=False, to_onehot_y=True)
        # include_background=False -> if False, channel index 0 (background category) is excluded from the calculation. if the non-background segmentations are small compared to the total image size they can get overwhelmed by the signal from the background so excluding it in such cases helps convergence.

        self.train_dsc = torchmetrics.Dice()
        self.validation_dsc = torchmetrics.Dice()

        self.learning_rate = 0.0001

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['target']
        batch_size = len(y)
        y_hat = self.unet(x)

        # y = torch.unsqueeze(y, dim=0)
        # y_hat = torch.unsqueeze(y_hat, dim=0)
        loss = self.criterion(y_hat, y)
        # input y_hat (Tensor) – the shape should be BNH[WD], where N is the number of classes.
        # target  y (Tensor) – the shape should be BNH[WD] or B1H[WD], where N is the number of classes.
        # Got: torch.Size([1, 1, 124, 256, 256])
        
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)

        self.train_dsc(y_hat, y)
        self.log('train_dsc', self.train_dsc,
                 on_step=True,  on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['target']

        batch_size = len(y)
        y_hat = self.unet(x)
        # y = torch.unsqueeze(y, dim=0)
        # y_hat = torch.unsqueeze(y_hat, dim=0)
        
        print(y_hat.shape, y.shape)
        loss = self.criterion(y_hat, y)

        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)

        self.validation_dsc(y_hat, y)
        self.log('validation_dsc', self.validation_dsc,
                 on_step=True,  on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = ReduceLROnPlateau(optimizer, 'min',
                                factor=0.1, patience=10)
        # learning rate scheduler
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch,
                                 "monitor": "val_loss"}
                }
