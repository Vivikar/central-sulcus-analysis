import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.unet3d.model_encoders import UNet3D


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

        # Base model f(.) output of last downsample layer
        self.unet_encoder = encoder

        # extract information about last layer embedding dimension
        number_of_layers = len(self.unet_encoder.f_maps)
        last_layer_kernels = self.unet_encoder.f_maps[-1]

        # calculate u-net embeding dimension
        embed_dim = int(last_layer_kernels * ((img_dim/(2)**(number_of_layers-1))**3))

        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.mlp_head = nn.Sequential(self.unet_encoder,
                                      nn.Flatten(),
                                      nn.ReLU(inplace=False),
                                      nn.Linear(embed_dim, hidden_dim))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode='train'):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)
        # Encode all images
        feats = self.mlp_head(imgs)

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)

        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)

        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode+'_loss', nll)

        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        # Logging ranking metrics
        self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean(),
                 prog_bar=True)
        self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean(),
                 prog_bar=True)
        self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode='val')
