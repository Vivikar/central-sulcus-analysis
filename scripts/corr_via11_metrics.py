# %%
# %%
# %%
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pprint import pprint

from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import sys
from pathlib import Path

import torch.nn as nn
import hydra

import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule

import src.utils.default as utils
sitk.ProcessObject_SetGlobalWarningDisplay(False)
from src.data.bvisa_dm import CS_Dataset

import matplotlib.pyplot as plt

from monai.metrics import compute_dice, compute_hausdorff_distance, compute_iou

import os
from src.data.splits import (bvisa_splits, bvisa_left_sulci_labels,
                             bvisa_right_sulci_labels, bvisa_padding_dims,
                             bad_via11, via11_qc)
from src.utils.general import crop_image_to_content, resample_volume, post_prcosess_segm
from src.utils.general import sitk_cropp_padd_img_to_size
from skimage.morphology import binary_dilation, label
class CS_eval_via(CS_Dataset):
    def __init__(self,
                 corrections_path: str,
                 resample: list[float] | None = None,
                 crop2content: bool = False,
                 croppadd2same_size: str = None):
        """Constructor for CS_Dataset class

        Args:
            corrections_path (str): Path to corrected segmentations.
            resample (list[x, y, z] | None, optional): Resample the images to a given resolution.
            crop2content (bool, optional): Crop the images to the content of the image.
            padd2same_size (string, optional): Pad-cropps the images to the same size depending on image type.
        """

        # save dataset hyperparameters
        self.corrections_path = Path(corrections_path)
        self.resample = list(resample) if resample is not None else None
        self.crop2content = crop2content
        self.croppadd2same_size = croppadd2same_size
        if self.croppadd2same_size:
            self.cropPadd_size = [int(x) for x in croppadd2same_size.split('-')]
        # load corresponding image and target paths
        self.img_paths = []
        self.target_paths = []
        self.corrected_target_paths = []
        self.caseids = []

        self._load_via11_corrected()

    def _load_via11_corrected(self):
        """Load VIA11 dataset"""
        drcmr_path = Path(os.environ.get('VIA11_DRCMR'))
        cfin_path = Path(os.environ.get('VIA11_CFIN'))

        cfin_subjs = [subj for subj in cfin_path.iterdir() if subj.is_dir()]
        drcmr_subjs = [subj for subj in drcmr_path.iterdir() if subj.is_dir()]
        all_subjs = cfin_subjs + drcmr_subjs
        
        # filter only corrected subjects

        cfin_corr = set([str(x.name).removeprefix('LSulci_').removeprefix('RSulci_')[:10] for x in (self.corrections_path/'CFIN').glob('*.nii.gz')])
        drcmr_corr = set([str(x.name).removeprefix('LSulci_').removeprefix('RSulci_')[:10] for x in (self.corrections_path/'DRCMR').glob('*.nii.gz')])
        all_corr = cfin_corr.union(drcmr_corr)
        all_subjs = [subj for subj in all_subjs if subj.name in all_corr]
        
        for subj in all_subjs:
            self.__load_subject_via11(subj)


    def __load_subject_via11(self, subj):
        sulci_path = os.environ.get('SEGM_PATH')

        # get image input paths
        image_paths = []

        image_paths.append(subj/f't1mri/default_acquisition/default_analysis/segmentation/skull_stripped_{subj.name}.nii.gz')

        subj_id = subj.name
        self.caseids.append(subj_id)
        self.img_paths.append(image_paths)

        # get target paths
        target_paths = []
        target_paths.append(subj/f'{sulci_path}/LSulci_{subj_id}_default_session_best.nii.gz')
        target_paths.append(subj/f'{sulci_path}/RSulci_{subj_id}_default_session_best.nii.gz')
        self.target_paths.append(target_paths)
        
        # get corrected  target paths
        corr_paths = []
        
        pref = '/mnt/projects/VIA_Vlad/nobackup/BrainVisa/CS_edited'
        center = str(target_paths[0]).split('BrainVisa/BrainVisa/')[1].split('/')[0]
        subj = str(target_paths[0]).split(center)[1][1:11]
        corrp_L = f'{pref}/{center}/LSulci_{subj}_default_session_best_edit_NHT.nii.gz'
        corrp_R = f'{pref}/{center}/RSulci_{subj}_default_session_best_edit_NHT.nii.gz'
    
        corr_paths.append(corrp_L)
        corr_paths.append(corrp_R)
        self.corrected_target_paths.append(corr_paths)
        

    def _load_image_target(self, idx):
        # load image
        image = sitk.ReadImage(str(self.img_paths[idx][0]))
        # reorient to RAS to have similar orientation as BVISA images
        image = sitk.DICOMOrient(image, 'RAS')
        
        # load target
        ltarget = sitk.ReadImage(str(self.target_paths[idx][0]))
        rtarget = sitk.ReadImage(str(self.target_paths[idx][1]))
        target = sitk.Cast((ltarget + rtarget) > 0, sitk.sitkInt16)
        # reorient to RAS to have similar orientation as BVISA images
        target = sitk.DICOMOrient(target, 'RAS')

        # load target
        ltarget = sitk.ReadImage(str(self.corrected_target_paths[idx][0]))
        rtarget = sitk.ReadImage(str(self.corrected_target_paths[idx][1]))
        corr_target = sitk.Cast((ltarget + rtarget) > 0, sitk.sitkInt16)
        # reorient to RAS to have similar orientation as BVISA images
        corr_target = sitk.DICOMOrient(corr_target, 'RAS')
        
        return image, target, corr_target

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
    
        image, target, corr_target = self._load_image_target(idx)

        # pre-process images
        image, target, corr_target = self._preprocess(image, target, corr_target)

        # post-process images
        image, target, corr_target = self._postprocess(image, target, corr_target)

        sample = {'image': image, 'target': target, 'corr_target': corr_target}

        # get caseid
        sample['caseid'] = self.caseids[idx]

        return sample

    def _preprocess(self, image, target, corr_target):
        """Converts from sitk.Image to torch.Tensor and
           ensures that targets have proper labels and
           images are normalized.

        Args:
            image (sitk.Image): Input image.
            target (sitk.Image): Target image.

        Returns:
            tuple(Tensor, Tensor): Pre-processed image and target.
        """
        # resample if needed
        if self.resample is not None:
            image_interpolator = sitk.sitkLinear
            image = resample_volume(image, self.resample, image_interpolator)
            target = resample_volume(target, self.resample,
                                     interpolator=sitk.sitkNearestNeighbor)
            corr_target = resample_volume(corr_target, self.resample,
                                     interpolator=sitk.sitkNearestNeighbor)
        
        # crop to content
        if self.crop2content:
            img2crop = sitk.GetArrayFromImage(image)
            target2crop = sitk.GetArrayFromImage(target)
            coorrtarget2crop = sitk.GetArrayFromImage(corr_target)
            image, min_coords, max_coords = crop_image_to_content(img2crop)
            target, _, __ = crop_image_to_content(target2crop, min_coords, max_coords)
            corr_target, _, __ = crop_image_to_content(coorrtarget2crop, min_coords, max_coords)
            image = sitk.GetImageFromArray(image)
            target = sitk.GetImageFromArray(target)
            corr_target = sitk.GetImageFromArray(corr_target)

        if self.croppadd2same_size:
            # cropp-padding if needed
            image = sitk_cropp_padd_img_to_size(image, self.cropPadd_size)
            target = sitk_cropp_padd_img_to_size(target, self.cropPadd_size)
            corr_target = sitk_cropp_padd_img_to_size(corr_target, self.cropPadd_size)

        # convert to numpy
        image = sitk.GetArrayFromImage(image)
        target = sitk.GetArrayFromImage(target)
        corr_target = sitk.GetArrayFromImage(corr_target)


        # min-max normalization of the image
        image = (image - image.min()) / (image.max() - image.min())

        return torch.Tensor(image), torch.tensor(target, dtype=torch.long), torch.tensor(corr_target, dtype=torch.long)

    def _postprocess(self, image: torch.Tensor, target: torch.Tensor, corr_target):
        # padd if needed
        # TODO: FIX THE ERROR WITH THE ACTIVE PADDING
        # if self.padd2same_size:
        #     raise ValueError('Should not be used anymore')
        #     size_key = 'original' if self.resample is None else str(self.resample)
        #     pad_dims = bvisa_padding_dims[self.input][size_key]
        #     padd = SpatialPad(pad_dims, mode='constant', value=0)
        #     image = padd(torch.unsqueeze(image, dim=0))[0]
        #     target = padd(torch.unsqueeze(target, dim=0))[0]

        # add channel dimension to the image
        image = torch.unsqueeze(image, 0)
        return image, target, corr_target

# %%


# %%
CHKP = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/logs_finetuning/CS1x_via11simBVISASST_tverskyLoss_monaBasicUnet-fullFinetune-MaxPool/runs/2023-04-27_15-07-50/checkpoints/epoch-106-Esubj-0.4295.ckpt')

print(CHKP)
out_path = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/data/via11/nobackup/segm_results/skull_stripped_images')

corrected_path = '/mnt/projects/VIA_Vlad/nobackup/BrainVisa/CS_edited'

exp_name = CHKP.parent.parent.parent.parent.name
out_path = out_path / exp_name
out_path.mkdir(exist_ok=True, parents=True)
cgf_path = CHKP.parent.parent / '.hydra' / 'config.yaml'
finetune_cfg = OmegaConf.load(cgf_path)

segm_model: LightningModule = hydra.utils.instantiate(finetune_cfg.model)
# sst_ds = hydra.utils.instantiate(finetune_cfg.data)
# print(finetune_cfg.data)
segm_model = segm_model.load_from_checkpoint(CHKP).to('cuda')
segm_model = segm_model.eval()


# %% [markdown]
# # Load via validation images

# %% USE ONLY FOR 1X DATASET
finetune_cfg.data.dataset_cfg.resample = [1, 1, 1.4]

WRITE_P = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/data/via11/nobackup/segm_results/correctes_res')

# %%
croppadd2same_size =  finetune_cfg.data.dataset_cfg.get('padd2same_size') if finetune_cfg.data.dataset_cfg.get('padd2same_size') else finetune_cfg.data.dataset_cfg.get('croppadd2same_size')

via11DS = CS_eval_via(corrected_path,
                      resample=finetune_cfg.data.dataset_cfg.resample,
                      croppadd2same_size=croppadd2same_size,
                      crop2content=True,)

# %%
experiment_results = []
for idx in tqdm(range(len(via11DS))):
    sample = via11DS[idx]
    target_1hot = torch.nn.functional.one_hot(sample['target'].unsqueeze(0), num_classes=2).permute(0, 4, 1, 2, 3)
    corrected_target_1hot = torch.nn.functional.one_hot(sample['corr_target'].unsqueeze(0), num_classes=2).permute(0, 4, 1, 2, 3)
    with torch.no_grad():
        out = segm_model.forward(sample['image'].unsqueeze(0).to('cuda'))
    # out_bin = torch.argmax(torch.softmax(out, dim=1), dim=1).cpu()
    # print(out_bin.shape, out_bin.unique(), out_bin.dtype, out_bin.max(), out_bin.min(), out_bin.sum())

    # apply post-processing
    segm_pred_bin = torch.softmax(out.cpu(), dim=1)[:, 0, :, :, :].squeeze(0).squeeze(0).numpy()
    segm_pred_bin = np.logical_not(segm_pred_bin > 0.5).astype(np.int16)
    segm_pred_bin = post_prcosess_segm(segm_pred_bin, dilations=0)
    out_bin = torch.tensor(segm_pred_bin, dtype=torch.int64, device='cpu').unsqueeze(0)
    # print(out_bin.shape, out_bin.unique(), out_bin.dtype, out_bin.max(), out_bin.min(), out_bin.sum())

    img_sitk = sitk.GetImageFromArray(sample['image'][0].numpy().astype(np.float32))
    sitk.WriteImage(img_sitk, str(WRITE_P / f'images/{sample["caseid"]}.nii.gz'))

    segm_pred_bin = out_bin.squeeze(0).numpy().astype(np.int16)
    pred_img = sitk.GetImageFromArray(segm_pred_bin)
    sitk.WriteImage(pred_img, str(WRITE_P / f'predicted/segmentations/{sample["caseid"]}.nii.gz'))

    corrected_bin = sample['corr_target'].numpy().astype(np.int16)
    pred_img = sitk.GetImageFromArray(corrected_bin)
    sitk.WriteImage(pred_img, str(WRITE_P / f'corrected/segmentations/{sample["caseid"]}.nii.gz'))

    bvisa_bin = sample['target'].numpy().astype(np.int16)
    pred_img = sitk.GetImageFromArray(bvisa_bin)
    sitk.WriteImage(pred_img, str(WRITE_P / f'brainvisa/segmentations/{sample["caseid"]}.nii.gz'))

    out_1hot = torch.nn.functional.one_hot(out_bin, num_classes=2).permute(0, 4, 1, 2, 3)

    pred_dice = compute_dice(out_1hot, corrected_target_1hot, include_background=False)
    pred_iou = compute_iou(out_1hot, corrected_target_1hot, include_background=False)
    pred_hausdorff_distance = compute_hausdorff_distance(out_1hot, corrected_target_1hot,
                                                    include_background=False)

    bvisa_dice = compute_dice(target_1hot, corrected_target_1hot, include_background=False)
    bvisa_iou = compute_iou(target_1hot, corrected_target_1hot, include_background=False)
    bvisa_hausdorff_distance = compute_hausdorff_distance(target_1hot, corrected_target_1hot,
                                                    include_background=False)
    
    res = {'caseid': via11DS.caseids[idx],
            'bvisa_dice':bvisa_dice.item(),
            'bvisa_iou':bvisa_iou.item(),
            'bvisa_hausdorff_distance':bvisa_hausdorff_distance.item(),
            'pred_dice':pred_dice.item(),
            'pred_iou':pred_iou.item(),
            'pred_hausdorff_distance':pred_hausdorff_distance.item(),
           }
    experiment_results.append(res)
experiment_results = pd.DataFrame(experiment_results)
experiment_results = experiment_results.set_index('caseid')
experiment_results.loc['MEAN'] = experiment_results.mean()
experiment_results.loc['STD'] = experiment_results.std()
experiment_results.to_csv(f'{out_path}/corr_via11_metrics.csv')

pprint(experiment_results)

