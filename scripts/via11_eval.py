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

CHKP = '/mrhome/vladyslavz/git/central-sulcus-analysis/logs_finetuning/CS1x_noSST_tverskyLoss_monaBasicUnet-fullTraining/runs/2023-04-12_13-42-02/checkpoints/epoch-045-Esubj-0.4378.ckpt'



out_path = '/mrhome/vladyslavz/git/central-sulcus-analysis/data/via11/nobackup/segm_results'

CHKP = Path(CHKP)
exp_name = CHKP.parent.parent.parent.parent.name
cgf_path = CHKP.parent.parent / '.hydra' / 'config.yaml'
segm_cfg = OmegaConf.load(cgf_path)

segm_model: LightningModule = hydra.utils.instantiate(segm_cfg.model)


segm_model = segm_model.load_from_checkpoint(CHKP).to('cuda')

via11DS = CS_Dataset('via11', 'mp2rage_raw',
                     'bvisa_CS', dataset_path='',
                     preload=False)

# via11DL = DataLoader(via11DS, batch_size=1, shuffle=False, num_workers=10)

for i in tqdm(range(len(via11DS))):
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

    res_path = Path(f'{out_path}/{exp_name}')
    res_path.mkdir(parents=True, exist_ok=True)

    # re-orient to PSR (original VIA orientation)
    segm_pred_sitk = sitk.DICOMOrient(segm_pred_sitk, 'ASL')
    orig_img = sitk.ReadImage(str(via11DS.img_paths[i][0]))

    try:
        segm_pred_sitk.CopyInformation(orig_img)
    except RuntimeError as e:
        print(f'Error: {e} at {caseid}')
    sitk.WriteImage(segm_pred_sitk, str(res_path/f'{caseid}.nii.gz'))
    # break
