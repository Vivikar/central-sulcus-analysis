from pathlib import Path
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


import hydra

import numpy as np
import torch
from omegaconf import  OmegaConf
from pytorch_lightning import LightningModule

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

CHKP = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/logs_finetuning/CS1x_noSST_noSynthAugm_monaiUnet/runs/2023-04-26_12-07-40/checkpoints/epoch-077-Esubj-0.2938.ckpt')

SUBJ2GENERATE = ['sub-via052', 'sub-via320', 'sub-via052', 'sub-via273']


out_path = '/mrhome/vladyslavz/git/central-sulcus-analysis/data/via11/nobackup/segm_results/skull_stripped_images'

CHKP = Path(CHKP)
exp_name = CHKP.parent.parent.parent.parent.name
cgf_path = CHKP.parent.parent / '.hydra' / 'config.yaml'
segm_cfg = OmegaConf.load(cgf_path)

segm_model: LightningModule = hydra.utils.instantiate(segm_cfg.model)

croppadd2same_size =  segm_cfg.data.dataset_cfg.get('padd2same_size') if segm_cfg.data.dataset_cfg.get('padd2same_size') else segm_cfg.data.dataset_cfg.get('croppadd2same_size')
segm_model = segm_model.load_from_checkpoint(CHKP).to('cuda')
segm_cfg.data.dataset_cfg.resample = [1, 1, 1.4]
via11DS = CS_Dataset('via11', 'mp2rage_skull_stripped',
                     'bvisa_CS', dataset_path='',
                      crop2content=True,
                      split='only_good',
                      preload=False,
                      resample=segm_cfg.data.dataset_cfg.resample,
                      croppadd2same_size=croppadd2same_size)

# via11DL = DataLoader(via11DS, batch_size=1, shuffle=False, num_workers=10)

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

    segm_pred_bin = torch.softmax(segm_pred, dim=1)[:, 0, :, :, :].squeeze(0).squeeze(0).numpy()
    segm_pred_bin = np.logical_not(segm_pred_bin > 0.5).astype(np.int16)

    segm_pred_sitk = sitk.GetImageFromArray(segm_pred_bin)
    target_img = sitk.GetImageFromArray(target.numpy().astype(np.uint8))
    orig_img = sitk.GetImageFromArray(img[0].numpy().astype(np.float32))

    res_path = Path(f'{out_path}/{exp_name}/{caseid}')
    res_path.mkdir(parents=True, exist_ok=True)

    # re-orient to PSR (original VIA orientation)
    # segm_pred_sitk = sitk.DICOMOrient(segm_pred_sitk, 'ASL')
    # orig_img = sitk.ReadImage(str(via11DS.img_paths[i][0]))

    sitk.WriteImage(segm_pred_sitk, str(res_path/f'prediction.nii.gz'))
    sitk.WriteImage(orig_img, str(res_path/f'image.nii.gz'))
    sitk.WriteImage(target_img, str(res_path/f'target.nii.gz'))
