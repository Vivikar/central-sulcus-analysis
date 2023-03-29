import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.networks.nets.resnet import resnet18

from src.models.unet3d.model_encoders import UNet3D
from torchmetrics.functional import pairwise_cosine_similarity

from lightly.data import LightlyDataset, SimCLRCollateFunction
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead


class SimCLR(pl.LightningModule):

    def __init__(self,
                 hidden_dim: int,
                 lr: float,
                 max_epochs: int,
                 img_dim: int,
                 temperature: float,
                 weight_decay: float,
                 encoder: UNet3D,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 ):
        """_summary_

        Args:
            hidden_dim (int): number of neurons in the MLP head output aka dimensionality
                of feature vector
            lr (float): _description_
            img_dim (int): _description_
            temperature (float): _description_
            weight_decay (float): _description_
            encoder (UNet3D): _description_
            optimizer (torch.optim.Optimizer): _description_
            scheduler (torch.optim.lr_scheduler): _description_
        """
        super().__init__()
        self.save_hyperparameters(ignore=['net'])

        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'

        # extract information about last layer embedding dimension
        number_of_layers = len(encoder.f_maps)
        last_layer_kernels = encoder.f_maps[-1]

        # calculate u-net embeding dimension after max pooling
        embed_dim = int(last_layer_kernels * ((img_dim/(2)**(number_of_layers - 1))**3)/(2**3))

        self.learning_rate = lr

        self.mlp_head = nn.Sequential(encoder,
                                      nn.MaxPool3d(kernel_size=2, stride=2),
                                      nn.Flatten(),
                                      nn.Linear(embed_dim, embed_dim),
                                      nn.ReLU(inplace=False),
                                      nn.Linear(embed_dim, 256))

        self.criterion = NTXentLoss()

    def configure_optimizers(self):
        optimizer =  torch.optim.SGD(self.parameters(), lr=0.06)
        return optimizer

    def do_step(self, batch, mode='train'):
        imgs, _ = batch
        imgs1, imgs2 = imgs
        z0 = self.mlp_head(imgs1)
        z1 = self.mlp_head(imgs2)
        loss = self.criterion(z0, z1)
        self.log(mode+'_loss', loss, prog_bar=True)

        top1, top5 = self.accuracy(z0, z1)
        self.log(mode+'_acc_top1', top1,
                 prog_bar=True)
        self.log(mode+'_acc_top5', top5,
                 prog_bar=True)
        return loss

    def accuracy(self, v1, v2):
        sim_mat = pairwise_cosine_similarity(v1, v2)

        arg_sim_mat = sim_mat.argsort(dim=1, descending=True)
        top1_acc = (arg_sim_mat[:, 0] == torch.arange(0, arg_sim_mat.shape[0], device=sim_mat.device)).sum().float() / arg_sim_mat.shape[0]
        top5_acc = (arg_sim_mat[:, :5] == torch.arange(0, arg_sim_mat.shape[0], device=sim_mat.device).unsqueeze(1)).sum().float() / arg_sim_mat.shape[0]
        return top1_acc, top5_acc

    def training_step(self, batch, batch_idx):
        return self.do_step(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, mode='val')
