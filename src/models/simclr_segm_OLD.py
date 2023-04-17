import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from monai.networks.nets.resnet import resnet18
from monai.losses import DiceLoss
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
                 dice_weight: float = 0.5,
                 ):
        """_summary_

        Args:
            hidden_dim (int): number of neurons in the MLP head output aka dimensionality
                of feature vector
            lr (float): _description_
            img_dim (int): _description_
            temperature (float): _description_
            dice_weight (float): Weight of the dice term in the loss function.
                Between 0 and 1. Default: 0.5.
            weight_decay (float): _description_
            encoder (UNet3D): _description_
            optimizer (torch.optim.Optimizer): _description_
            scheduler (torch.optim.lr_scheduler): _description_
        """
        super().__init__()
        self.save_hyperparameters(ignore=['net', 'encoder'])

        self.dice_weight = dice_weight
        self.learning_rate = lr

        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'

        # extract information about last layer embedding dimension
        number_of_layers = len(encoder.f_maps)
        last_layer_kernels = encoder.f_maps[-1]

        # calculate u-net embeding dimension after max pooling
        embed_dim = int(last_layer_kernels * ((img_dim/(2)**(number_of_layers - 1))**3)/(2**3))

        self.unet = encoder

        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.mlp_head = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2),
                                      nn.Flatten(),
                                      nn.Linear(embed_dim, embed_dim),
                                      nn.ReLU(inplace=False),
                                      nn.Linear(embed_dim, hidden_dim))

        # define the loss functions for segmentation and contrastive learning
        self.contrastive_loss = torch.nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True,
                                  include_background=False)

        self.val_dsc = torchmetrics.Dice(ignore_index=0,
                                         average='macro',
                                         num_classes=3,
                                         multiclass=True)
        self.train_dsc = torchmetrics.Dice(ignore_index=0,
                                           average='macro',
                                           num_classes=3,
                                           multiclass=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def info_nce_loss(self, batch, mode='train'):

        # NEW UN-BATCHING WITH LABELS
        img1, img2, lab1, lab2 = batch
        pred_segm, bottle_neck_features = self.unet(torch.concat([img1, img2]))
        embeddings = self.mlp_head(bottle_neck_features)
        ########### InfoNCE Loss ###########

        # Encode all images
        # features = self.mlp_head(imgs)

        labels = torch.cat([torch.arange(batch[-1].shape[0], device=embeddings.device) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        embeddings = F.normalize(embeddings, dim=1)

        similarity_matrix = torch.matmul(embeddings, embeddings.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=similarity_matrix.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        logits = logits / self.hparams.temperature

        contrastive_loss = self.contrastive_loss(logits, labels)


        ### DICE Loss ###
        # pred_segm = self.unet(torch.concat([img1, img2]))

        dice_loss = self.dice_loss(pred_segm, torch.concat([lab1, lab2]))

        final_loss = (1-self.dice_weight)*contrastive_loss + self.dice_weight*dice_loss

        # Logging loss
        self.log(mode+'_final_loss', final_loss, prog_bar=True)
        self.log(mode+'_dice_loss', dice_loss, prog_bar=True)
        self.log(mode+'_contrastive_loss', contrastive_loss, prog_bar=True)
        
        top1, top5 = self.accuracy(logits, labels, topk=(1, 5))
        # # Get ranking position of positive example

        # Logging ranking metrics
        self.log(mode+'_acc_top1', top1[0],
                 prog_bar=True)
        self.log(mode+'_acc_top5', top5[0],
                 prog_bar=True)

        if mode == 'val':
            self.val_dsc(pred_segm,
                         torch.concat([lab1.squeeze(dim=1),
                                       lab2.squeeze(dim=1)]).type(torch.long))
            self.log(mode+'_dice', self.val_dsc.compute(), prog_bar=True,
                     on_step=True, on_epoch=True, batch_size=8, logger=True)
        elif mode == 'train':
            self.train_dsc(pred_segm,
                           torch.concat([lab1.squeeze(dim=1),
                                         lab2.squeeze(dim=1)]).type(torch.long))
            self.log(mode+'_dice', self.train_dsc.compute(), prog_bar=True,
                     on_step=True, on_epoch=True, batch_size=16, logger=True)
        return final_loss

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='val')
