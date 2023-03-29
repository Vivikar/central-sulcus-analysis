import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.networks.nets.resnet import resnet18

from src.models.unet3d.model_encoders import UNet3D


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
        optimizer = torch.optim.SGD(self.parameters(), lr=0.06)
        return optimizer

    def do_step(self, batch, mode='train'):
        imgs, _ = batch
        imgs1, imgs2 = imgs
        z0 = self.mlp_head(imgs1)
        z1 = self.mlp_head(imgs2)
        loss = self.criterion(z0, z1)
        self.log(mode+'_loss', loss, prog_bar=True)

        return loss

    #     return optimizer

    # def forward(self, x):
    #     return self.mlp_head(x)

    # def info_nce_loss(self, batch, mode='train'):
    #     imgs, _ = batch
    #     imgs = torch.cat(imgs, dim=0)
    #     # Encode all images
    #     features = self.mlp_head(imgs)
        
    #     labels = torch.cat([torch.arange(batch[-1].shape[0], device=features.device) for i in range(2)], dim=0)
    #     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    #     features = F.normalize(features, dim=1)
        
    #     similarity_matrix = torch.matmul(features, features.T)
        
    #     # discard the main diagonal from both: labels and similarities matrix
    #     mask = torch.eye(labels.shape[0], dtype=torch.bool, device=similarity_matrix.device)
    #     labels = labels[~mask].view(labels.shape[0], -1)
    #     similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    #     # assert similarity_matrix.shape == labels.shape

    #     # select and combine multiple positives
    #     positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    #     # select only the negatives the negatives
    #     negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    #     logits = torch.cat([positives, negatives], dim=1)
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    #     logits = logits / self.hparams.temperature
        
    #     loss = self.criterion(logits, labels)

    #     # # Calculate cosine similarity
    #     # cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

    #     # # Mask out cosine similarity to itself
    #     # self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    #     # cos_sim.masked_fill_(self_mask, -9e15)

    #     # # Find positive example -> batch_size//2 away from the original example
    #     # pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)

    #     # # InfoNCE loss
    #     # cos_sim = cos_sim / self.hparams.temperature
    #     # nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    #     # nll = nll.mean()

    #     # Logging loss
    #     self.log(mode+'_loss', loss, prog_bar=True)
    #     top1, top5 = self.accuracy(logits, labels, topk=(1, 5))
    #     # # Get ranking position of positive example
    #     # comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
    #     #                       cos_sim.masked_fill(pos_mask, -9e15)],
    #     #                      dim=-1)
    #     # sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

    #     # # Logging ranking metrics
    #     self.log(mode+'_acc_top1', top1[0],
    #              prog_bar=True)
    #     self.log(mode+'_acc_top5', top5[0],
    #              prog_bar=True)
    #     # self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())
    #     return loss
    #     # return nll
    # @staticmethod
    # def accuracy(output, target, topk=(1,)):
    #     """Computes the accuracy over the k top predictions for the specified values of k"""
    #     with torch.no_grad():
    #         maxk = max(topk)
    #         batch_size = target.size(0)

    #         _, pred = output.topk(maxk, 1, True, True)
    #         pred = pred.t()
    #         correct = pred.eq(target.view(1, -1).expand_as(pred))

    #         res = []
    #         for k in topk:
    #             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    #             res.append(correct_k.mul_(100.0 / batch_size))
    #         return res
    def training_step(self, batch, batch_idx):
        return self.do_step(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, mode='val')
