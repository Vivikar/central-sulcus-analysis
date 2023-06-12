import monai
import pytorch_lightning as pl
import torch
import torchmetrics
from kornia.losses import HausdorffERLoss3D
from monai.networks.nets import BasicUNet
from torchmetrics import MaxMetric, MeanMetric

from src.models.losses.dice import SoftDiceLoss
from src.models.monai_impl import BasicUNetEncoder
from src.models.simclr import SimCLR
from src.models.unet3d.model import UNet3D
from src.utils.metrics.error_rate import SulciErrorLocal, SulciErrorSubject


class RegrUNet3D(pl.LightningModule):
    """U-Net 3D model for regression tasks.
    Used in experiments with predicting the continuous sulci localization"""
    def __init__(
            self,
            lr: float,
            net: UNet3D,
            out_channels: int,
            freeze_encoder: bool,
            monai: bool,
            loss_function,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            encoder_chkp: str | None = None,
            extra_loss: str | None = None,
            ):

        super().__init__()

        self.loss_function= loss_function
        self.net = net
        self.encoder_chkp = encoder_chkp
        self.monai = monai
        self.freeze_encoder = freeze_encoder
        self.extra_loss = extra_loss

        # pre-load weights
        if self.encoder_chkp is not None:
            print('Loading encoder weights from checkpoint...')
            print(self.encoder_chkp)
            simclr = SimCLR.load_from_checkpoint(self.encoder_chkp,
                                                 strict=False)
            if not monai:
                # load pretrained silclr encoder from the custom UNET
                # replace the encoder with the pretrained one
                self.net.encoders = simclr.mlp_head[0].encoders
            else:
                self.net.conv_0 = simclr.encoder.conv_0
                self.net.down_1 = simclr.encoder.down_1
                self.net.down_2 = simclr.encoder.down_2
                self.net.down_3 = simclr.encoder.down_3
                self.net.down_4 = simclr.encoder.down_4
        if self.freeze_encoder:
            print('Freezing encoder...')
            i = 0
            for child in self.net.children():
                if i > 4:
                    break
                i += 1
                for param in child.parameters():
                    param.requires_grad = False

        # loss function
        self.criterion = loss_function
        self.learning_rate = lr
        self.out_channels = out_channels
        
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()
        self.save_hyperparameters()

    def forward(self, x):
        if not self.monai:
            return self.net.forward(x)
        else:
            return self.net(x)

    def _on_step(self, batch, batch_idx):
        x = batch['image']  # (batch_size, 1, 128, 128, 128)
        y = batch['target']  # (batch_size, 128, 128, 128)
        if not self.monai:
            y_hat = self.net.forward(x)  # (batch_size, num_classes, 128, 128, 128)
        else:
            y_hat = self.net(x).squeeze(dim=0)  # (batch_size, num_classes, 128, 128, 128)

        return y, y_hat, len(y)

    def _get_loss(self, input, target, step=None):

        segm_loss = None
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            # for cross entropy loss 
            # input shape: (batch_size, num_classes, 128, 128, 128)
            # target shape: (batch_size, 128, 128, 128)
            segm_loss =  self.criterion(input, target)

        elif isinstance(self.criterion, monai.losses.DiceLoss) or\
             isinstance(self.criterion, monai.losses.DiceCELoss) or\
             isinstance(self.criterion, monai.losses.FocalLoss) or\
             isinstance(self.criterion, monai.losses.TverskyLoss):
            # for dice loss
            # input shape: (batch_size, num_classes, 128, 128, 128)
            # target shape: (batch_size, 1 or num_classes, 128, 128, 128)
            segm_loss = self.criterion(input, torch.unsqueeze(target, dim=1))

        elif isinstance(self.criterion, monai.losses.MaskedDiceLoss):

            segm_loss =  self.criterion(input,
                                  torch.unsqueeze(target, dim=1),
                                  mask=input)
        elif isinstance(self.criterion, SoftDiceLoss):
            segm_loss =  self.criterion(input, target)
        elif isinstance(self.criterion, torch.nn.MSELoss):
            segm_loss =  self.criterion(input, target)
        elif isinstance(self.criterion, torch.nn.BCELoss):
            segm_loss =  self.criterion(input, target)
        elif isinstance(self.criterion, torch.nn.L1Loss):
            segm_loss =  self.criterion(input, target)
        else:
            raise ValueError("Loss function not supported")

        if self.extra_loss == 'hausdorff':
            hd_loss = self.hausdorff(input, torch.unsqueeze(target, dim=1))
            if step == 'val':
                self.log("train/loss", hd_loss, on_step=False,
                         on_epoch=True, prog_bar=True,
                         logger=True, batch_size=1)
            total_loss = segm_loss + hd_loss
        else:
            total_loss = segm_loss
        return total_loss

    def training_step(self, batch, batch_idx):

        target, input, batch_size = self._on_step(batch, batch_idx)

        loss = self._get_loss(input, target)

        self.log("train/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)

        train_mse = self.val_mse(input, target)
        self.log("train/mse", train_mse, on_step=True,
                 prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):

        target, input, batch_size = self._on_step(batch[0], batch_idx)

        loss = self._get_loss(input, target, 'val')

        self.log("val/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)

        val_mse = self.val_mse(input, target)
        self.log("val/mse", val_mse, on_step=False)

        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters(), lr=self.learning_rate)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
