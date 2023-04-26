import pytorch_lightning as pl
import monai
import torch
import torchmetrics
from monai.networks.nets import BasicUNet
from kornia.losses import HausdorffERLoss3D
from torchmetrics import MaxMetric, MeanMetric
from src.models.simclr import SimCLR
from src.models.simclr_segm import SimCLRSegm
from src.models.unet3d.model import UNet3D
from src.utils.metrics.error_rate import SulciErrorLocal, SulciErrorSubject
from src.models.monai_impl import BasicUNetEncoder

class BasicUNet3D(pl.LightningModule):
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
            segm_encoder: bool = False,
            ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.loss_function= loss_function
        self.net = net
        self.encoder_chkp = encoder_chkp
        self.monai = monai
        self.segm_encoder = segm_encoder
        self.freeze_encoder = freeze_encoder
        self.extra_loss = extra_loss
        # pre-load weights
        if self.encoder_chkp is not None:
            print('Loading encoder weights from checkpoint...')
            print(self.encoder_chkp)
            if segm_encoder:
                simclr = SimCLRSegm.load_from_checkpoint(self.encoder_chkp,
                                                 strict=True)
            else:
                simclr = SimCLR.load_from_checkpoint(self.encoder_chkp,
                                                 strict=True)
            if not monai:
                # load pretrained silclr encoder from the custom UNET
                # replace the encoder with the pretrained one
                self.net.encoders = simclr.mlp_head[0].encoders
            else:
                if segm_encoder:
                    self.net = simclr.unet
                else:
                    self.net.conv_0 = simclr.encoder.conv_0
                    self.net.down_1 = simclr.encoder.down_1
                    self.net.down_2 = simclr.encoder.down_2
                    self.net.down_3 = simclr.encoder.down_3
                    self.net.down_4 = simclr.encoder.down_4
            # TODO: Implelent loading of downCONV part
        if self.freeze_encoder:
            print('Freezing encoder...')
            i = 0
            for child in self.net.children():
                if i > 4:
                    break
                i+=1
                for param in child.parameters():
                    param.requires_grad = False

        # loss function
        self.criterion = loss_function
        self.learning_rate = lr
        self.out_channels = out_channels
        avg_type = 'macro' if self.out_channels > 2 else 'micro'

        # metrics to track (ignore_index=0 is for background class)
        self.train_dsc = torchmetrics.Dice(ignore_index=0,
                                           average=avg_type,
                                           num_classes=self.out_channels)
        self.train_eloc = SulciErrorLocal(ignore_index=[0])
        self.train_esubj = SulciErrorSubject(ignore_index=[0])

        self.val_dsc = torchmetrics.Dice(ignore_index=0,
                                         average=avg_type,
                                         num_classes=self.out_channels)
        self.val_eloc = SulciErrorLocal(ignore_index=[0])
        self.val_esubj = SulciErrorSubject(ignore_index=[0])

        self.test_dsc = torchmetrics.Dice(ignore_index=0,
                                          average=avg_type,
                                          num_classes=self.out_channels)
        self.test_eloc = SulciErrorLocal(ignore_index=[0])
        self.test_esubj = SulciErrorSubject(ignore_index=[0])
        self.val_dsc_best = MaxMetric()

        # for averaging loss across batches
        self.test_loss = MeanMetric()

        if self.extra_loss == 'hausdorff':
            self.hausdorff = HausdorffERLoss3D()

        self.save_hyperparameters()

    def forward(self, x):
        if not self.monai:
            return self.net.forward(x)
        else:
            if self.segm_encoder:
                return self.net(x)[-1] # retrieve the segm not the embed
            else:
                return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_dsc_best.reset()

    def _on_step(self, batch, batch_idx):
        x = batch['image']  # (batch_size, 1, 128, 128, 128)
        y = batch['target']  # (batch_size, 128, 128, 128)
        if not self.monai:
            y_hat = self.net.forward(x)  # (batch_size, num_classes, 128, 128, 128)
        else:
            # (batch_size, num_classes, 128, 128, 128)
            if self.segm_encoder:
                y_hat = self.net(x)[-1] # retrieve the segm not the embed
            else:
                y_hat =  self.net(x)
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
            segm_loss =  self.criterion(input, torch.unsqueeze(target, dim=1))

        elif isinstance(self.criterion, monai.losses.MaskedDiceLoss):

            segm_loss =  self.criterion(input,
                                  torch.unsqueeze(target, dim=1),
                                  mask=input)
        else:
            raise ValueError("Loss function not supported")

        if self.extra_loss == 'hausdorff':
            hd_loss = self.hausdorff(input, torch.unsqueeze(target, dim=1))
            if step == 'val':
                self.log("train/loss", hd_loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 logger=True, batch_size=1)
            total_loss = segm_loss + hd_loss
        elif self.extra_loss is None:
            total_loss = segm_loss
        return total_loss

    def training_step(self, batch, batch_idx):

        target, input, batch_size = self._on_step(batch, batch_idx)

        loss = self._get_loss(input, target)

        self.log("train/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)

        self.train_dsc(input, target)
        self.log('train/dsc', self.train_dsc,
                 on_step=True,  on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=batch_size)

        # self.train_eloc(input, target)
        # self.log('train/Eloc', self.train_eloc,
        #          on_epoch=True, prog_bar=False,
        #          on_step=True, logger=True,
        #          batch_size=batch_size)

        self.train_esubj(input, target)
        self.log('train/Esubj', self.train_esubj,
                 on_epoch=True, prog_bar=False,
                 on_step=True, logger=True,
                 batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):

        target, input, batch_size = self._on_step(batch[0], batch_idx)

        loss = self._get_loss(input, target, 'val')

        self.log("val/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)

        self.val_dsc(input, target)
        self.log('val/dsc', self.val_dsc,
                 on_step=True,  on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=batch_size)

        # self.val_eloc(input, target)
        # self.log('val/Eloc', self.val_eloc,
        #          on_epoch=True, prog_bar=False,
        #          on_step=True, logger=True,
        #          batch_size=batch_size)

        self.val_esubj(input, target)
        self.log('val/Esubj', self.val_esubj,
                 on_epoch=True, prog_bar=True,
                 on_step=True, logger=True,
                 batch_size=batch_size)
        if batch[1]:
            self.test_step(batch[1], batch_idx)

        return loss

    def validation_epoch_end(self, outputs: list):
        # get current val dsc
        dsc = self.val_dsc.compute()

        # update best so far val dsc
        self.val_dsc_best(dsc)

        # log `val_dsc_best` as a value through `.compute()` method, instead
        # of as a metric object otherwise metric would be reset by lightning after each epoch
        self.log("val/dsc_best", self.val_dsc_best.compute(), prog_bar=True)


    def test_step(self, batch, batch_idx: int):
        target, input, batch_size = self._on_step(batch, batch_idx)

        loss = self._get_loss(input, target)

        # update and log metrics
        self.test_loss(loss)
        self.test_dsc(input, target)
        self.log("test/loss", self.test_loss, on_step=False,
                 on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test/dsc", self.test_dsc, on_step=False,
                 on_epoch=True, prog_bar=True, batch_size=batch_size)

        # self.test_eloc(input, target)
        # self.log('test/Eloc', self.test_eloc,
        #          on_epoch=True, prog_bar=False,
        #          on_step=True, logger=True,
        #          batch_size=batch_size)

        self.test_esubj(input, target)
        self.log('test/Esubj', self.test_esubj,
                 on_epoch=True, prog_bar=False,
                 on_step=True, logger=True,
                 batch_size=batch_size)

        # return loss

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
