from pathlib import Path
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


import hydra

import numpy as np
import torch
from omegaconf import  OmegaConf
from pytorch_lightning import LightningModule
from monai.metrics import compute_dice, compute_hausdorff_distance, compute_iou

from tqdm import tqdm

import src.utils.default as utils
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)
# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


log = utils.get_pylogger(__name__)

torch.set_float32_matmul_precision('medium')
# torch.autograd.set_detect_anomaly(True)

import SimpleITK as sitk

from src.data.bvisa_dm import CS_Dataset

sitk.ProcessObject_SetGlobalWarningDisplay(False)


CHKP_PATHS = [Path('/mrhome/vladyslavz/git/central-sulcus-analysis/logs_finetuning/CS1x_via11simBVISASST_tverskyLoss_monaBasicUnet-fullFinetune-MaxPool/runs/2023-04-27_15-07-50/checkpoints/epoch-106-Esubj-0.4295.ckpt'),
              ]

GOOD_subj_1 = ['sub-via052', 'sub-via320', 'sub-via325', 'sub-via273']
MEH_subj_2 = ['sub-via043', 'sub-via065', 'sub-via127', 'sub-via139', 'sub-via081', 'sub-via160', 'sub-via186', 'sub-via224']
BAD_subj_3 = ['sub-via135', 'sub-via191', 'sub-via311', 'sub-via365']



SUBJ2GENERATE = GOOD_subj_1 + MEH_subj_2 + BAD_subj_3

for CHKP in CHKP_PATHS:



    out_path = '/mrhome/vladyslavz/git/central-sulcus-analysis/data/via11/nobackup/segm_results/test_segm_results'

    CHKP = Path(CHKP)
    exp_name = CHKP.parent.parent.parent.parent.name
    cgf_path = CHKP.parent.parent / '.hydra' / 'config.yaml'
    segm_cfg = OmegaConf.load(cgf_path)

    segm_model: LightningModule = hydra.utils.instantiate(segm_cfg.model)

    croppadd2same_size =  segm_cfg.data.dataset_cfg.get('padd2same_size') if segm_cfg.data.dataset_cfg.get('padd2same_size') else segm_cfg.data.dataset_cfg.get('croppadd2same_size')
    segm_model = segm_model.load_from_checkpoint(CHKP).to('cuda')
    if '2x' in str(CHKP):
        segm_cfg.data.dataset_cfg.resample = [2, 2, 2]
    else:
        segm_cfg.data.dataset_cfg.resample = [1, 1, 1.4]

    via11DS = CS_Dataset('via11', 'mp2rage_skull_stripped',
                         'bvisa_CS', dataset_path='',
                         crop2content=True,
                         split=None,
                         preload=False,
                         resample=segm_cfg.data.dataset_cfg.resample,
                         croppadd2same_size=croppadd2same_size)


    INDX2GENERATE = []
    for idx, img_path in enumerate(via11DS.img_paths):
        for s2g in SUBJ2GENERATE:
            if s2g in img_path[0].name:
                INDX2GENERATE.append(idx)

    for i in tqdm(INDX2GENERATE):
        val_sample = via11DS[i]
        img = val_sample['image']
        target = val_sample['target']
        caseid = val_sample['caseid']

        with torch.no_grad():
            segm_pred = segm_model(img.unsqueeze(0).to('cuda')).to('cpu')

        target_1hot = torch.nn.functional.one_hot(target.unsqueeze(0), num_classes=2).permute(0, 4, 1, 2, 3)
        out_bin = torch.argmax(torch.softmax(segm_pred, dim=1), dim=1).cpu()
        out_1hot = torch.nn.functional.one_hot(out_bin, num_classes=2).permute(0, 4, 1, 2, 3)

        dice = compute_dice(out_1hot, target_1hot, include_background=False).item()
        hausdorff_distance = compute_hausdorff_distance(out_1hot, target_1hot, include_background=False).item()

        segm_pred_bin = torch.softmax(segm_pred, dim=1)[:, 0, :, :, :].squeeze(0).squeeze(0).numpy()
        segm_pred_bin = np.logical_not(segm_pred_bin > 0.5).astype(np.int16)

        segm_pred_sitk = sitk.GetImageFromArray(segm_pred_bin)
        target_img = sitk.GetImageFromArray(target.numpy().astype(np.uint8))
        orig_img = sitk.GetImageFromArray(img[0].numpy().astype(np.float32))

        prefix = 1 if caseid in GOOD_subj_1 else 2 if caseid in MEH_subj_2 else 3
        res_path = Path(f'{out_path}/{exp_name}/{prefix}-{caseid}-dice-{dice:.4f}-hausdorff-{hausdorff_distance:.4f}')
        res_path.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(segm_pred_sitk, str(res_path/f'prediction.nii.gz'))
        sitk.WriteImage(orig_img, str(res_path/f'image.nii.gz'))
        sitk.WriteImage(target_img, str(res_path/f'target.nii.gz'))
