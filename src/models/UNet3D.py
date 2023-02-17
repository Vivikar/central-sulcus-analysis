import pytorch_lightning as pl
import torch
import torchmetrics
from monai.networks.nets import BasicUNet
from torchmetrics import MaxMetric, MeanMetric


class BasicUNet3D(pl.LightningModule):
    def __init__(
            self,
            net: BasicUNet,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            loss_function):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=['net', 'loss_function'])

        self.net = net

        # loss function
        self.criterion = loss_function

        # metrics to track
        self.train_dsc = torchmetrics.Dice()
        self.val_dsc = torchmetrics.Dice()
        self.test_dsc = torchmetrics.Dice()

        self.val_dsc_best = MaxMetric()

        # for averaging loss across batches
        self.test_loss = MeanMetric()

    def forward(self, x):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_dsc_best.reset()

    def training_step(self, batch, batch_idx):

        x = batch['image']
        y = batch['target']

        batch_size = len(y)

        y_hat = self.net(x)

        loss = self.criterion(y_hat, y)

        self.log("train/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)

        self.train_dsc(y_hat, y)
        self.log('train/dsc', self.train_dsc,
                 on_step=True,  on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['target']

        batch_size = len(y)
        y_hat = self.net(x)

        loss = self.criterion(y_hat, y)

        self.log("val/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)

        self.val_dsc(y_hat, y)
        self.log('val/dsc', self.val_dsc,
                 on_step=True,  on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=batch_size)

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
        x = batch['image']
        y = batch['target']

        batch_size = len(y)
        y_hat = self.net(x)

        loss = self.criterion(y_hat, y)
        # update and log metrics
        self.test_loss(loss)
        self.val_dsc(y_hat, y)
        self.log("test/loss", self.test_loss, on_step=False,
                 on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test/dsc", self.test_dsc, on_step=False,
                 on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
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
