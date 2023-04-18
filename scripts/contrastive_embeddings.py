from pathlib import Path
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


import hydra

import numpy as np
import torch
from omegaconf import  OmegaConf
from pytorch_lightning import LightningModule

from tqdm import tqdm
import pandas as pd
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

CHKP = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/logs_sst/synthseg-monai-BasicUnet-1.5x-half/runs/2023-04-13_16-13-49/checkpoints/epoch-251_val_loss-0.000.ckpt')
out_path = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/data/via11/nobackup/contrastive_embeddings')

exp_name = CHKP.parent.parent.parent.parent.name
out_path = out_path / exp_name

cgf_path = CHKP.parent.parent / '.hydra' / 'config.yaml'
sst_cfg = OmegaConf.load(cgf_path)

sst_model: LightningModule = hydra.utils.instantiate(sst_cfg.model)


sst_model = sst_model.load_from_checkpoint(CHKP).to('cuda')

via11DS = CS_Dataset('via11', 'mp2rage_raw',
                     'bvisa_CS', dataset_path='',
                     preload=False,
                     resample=(1.5, 1.5, 1.5),
                     padd2same_size='171-171-171')

results = []

for i in tqdm(range(len(via11DS))):
    val_sample = via11DS[i]
    img = val_sample['image']
    target = val_sample['target']
    caseid = val_sample['caseid']

    with torch.no_grad():
        encoder_embed = sst_model.encoder(img.unsqueeze(0).to('cuda')).to('cpu')
        mlp_embed = sst_model(img.unsqueeze(0).to('cuda')).to('cpu')

    results.append({'caseid': caseid,
                    'encoder_embed': encoder_embed.numpy(),
                    'mlp_embed': mlp_embed.numpy(),
                    })

out_path.mkdir(parents=True, exist_ok=True)
results = pd.DataFrame(results)
results.to_csv(out_path / 'sst_embeds.csv')
results.to_pickle(out_path / 'sst_embeds.pkl')
