import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.networks.nets.resnet import resnet18
from monai.losses import contrastive
from src.models.unet3d.model_encoders import UNet3D
import torchmetrics

class SimCLRSegm(pl.LightningModule):

    def __init__(self,
                 hidden_dim: int,
                 lr: float,
                 max_epochs: int,
                 img_dim: int,
                 temperature: float,
                 weight_decay: float,
                 unet: UNet3D,
                 scheduler: torch.optim.lr_scheduler,
                 segm_loss,
                 dice_weight: float = 0.7,
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
            scheduler (torch.optim.lr_scheduler): _description_
            alpha: float: Contrastive loss weight
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dice_weight = dice_weight
        self.unet = unet
        self.save_hyperparameters()

        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'

        # extract information about last layer embedding dimension
        # number_of_layers = len(self.encoder.f_maps)
        # last_layer_kernels = self.encoder.f_maps[-1]

        # calculate u-net embeding dimension after max pooling
        # embed_dim = int(last_layer_kernels * ((img_dim/(2)**(number_of_layers - 1))**3)/(2**3))
        embed_dim = 131072    # TODO:  Hardcoded for now FOR BVISA DATA RETRAINING
        print(f'U-Net Embedding dimension: {embed_dim}')
        self.learning_rate = lr

        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.mlp_head = nn.Sequential(
                                      nn.Flatten(),

                                      nn.Linear(embed_dim, self.hidden_dim),
                                      nn.BatchNorm1d(self.hidden_dim),
                                      nn.ReLU(inplace=False),

                                      nn.Linear(self.hidden_dim, self.hidden_dim),
                                      nn.BatchNorm1d(self.hidden_dim),
                                      nn.ReLU(inplace=False),

                                      nn.Linear(self.hidden_dim, self.hidden_dim)
                                    )

        self.criterion = contrastive.ContrastiveLoss(temperature)  # torch.nn.CrossEntropyLoss()
        self.segm_loss = segm_loss

        # metrics to track (ignore_index=0 is for background class)
        self.train_dsc = torchmetrics.Dice(ignore_index=0, num_classes=2)
        self.val_dsc = torchmetrics.Dice(ignore_index=0, num_classes=2)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        unet_embed, unet_pred = self.unet(x)
        embed = self.mlp_head(unet_embed)
        return embed, unet_pred, 

    def info_nce_loss(self, batch, mode='train'):
        imgs, targets = batch
        imgs = torch.cat(imgs, dim=0)
        targets = torch.cat(targets, dim=0)

        # print(f'imgs shape: {imgs.shape}')
        # print(f'targets shape: {targets.shape}')
        features, segm_pred = self.forward(imgs)
        # print(f'features shape: {features.shape}')
        # ####### CONTRASTIVE LOSS ########
        batch_size = features.shape[0]//2
        contrastive_loss = self.criterion(features[:batch_size, :],
                                          features[batch_size:, :])

        # labels = torch.cat([torch.arange(len(imgs), device=features.device) for i in range(2)], dim=0)
        # labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # features = F.normalize(features, dim=1)

        # similarity_matrix = torch.matmul(features, features.T)

        # # discard the main diagonal from both: labels and similarities matrix
        # mask = torch.eye(labels.shape[0], dtype=torch.bool, device=similarity_matrix.device)
        # labels = labels[~mask].view(labels.shape[0], -1)
        # similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # # assert similarity_matrix.shape == labels.shape

        # # select and combine multiple positives
        # positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # # select only the negatives the negatives
        # negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # logits = torch.cat([positives, negatives], dim=1)
        # labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        # logits = logits / self.hparams.temperature

        # contrastive_loss = self.criterion(logits, labels)

        # Logging contrastive loss
        self.log(mode+'_loss_contr', contrastive_loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        # top1, top5 = self.accuracy(logits, labels, topk=(1, 2))

        # # # Logging ranking metrics
        # self.log(mode+'_acc_top1', top1[0],
        #          prog_bar=True, on_step=False, on_epoch=True)

        #### SEGMENTATION LOSS #######
        if isinstance(self.segm_loss, torch.nn.CrossEntropyLoss):
            segment_loss = self.segm_loss(segm_pred, targets)
        else:
            segment_loss = self.segm_loss(segm_pred,
                                          torch.unsqueeze(targets, dim=1))

        self.log(mode+'_loss_segm', segment_loss, prog_bar=True,
                 on_epoch=True, on_step=False)
        if mode == 'train':
            self.log(mode+'_dice', self.train_dsc(segm_pred, targets),
                     on_epoch=True, on_step=False, prog_bar=True)
        else:
            self.log(mode+'_dice', self.val_dsc(segm_pred, targets),
                     on_epoch=True, on_step=False, prog_bar=True)

        ######### TOTAL LOSS #########
        loss = (1-self.dice_weight)*contrastive_loss + self.dice_weight*segment_loss
        self.log(mode+'_loss', loss, prog_bar=True,
                 on_epoch=True, on_step=False)
        return loss

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
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='val')
