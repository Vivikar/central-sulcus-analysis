import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import sys
from pathlib import Path
from pprint import pprint

from pathlib import Path
import pyrootutils

import torch.nn as nn
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
from src.data.bvisa_dm import CS_Dataset

import SimpleITK as sitk

from src.data.bvisa_dm import CS_Dataset

import matplotlib.pyplot as plt

from monai.metrics import compute_dice, compute_hausdorff_distance, compute_iou

# %% [markdown]
# # Load segmentation model

# %%
CHKP = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/logs_finetuning/CS1x_noSST_SynthAugm_monaiUnet/runs/2023-04-26_12-18-13/checkpoints/epoch-100-Esubj-0.4202.ckpt')

out_path = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/data/brainvisa_results/nobackup')

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

print(f'Processing {exp_name}')

# %% [markdown]
# # Load via validation images

# %%
croppadd2same_size =  finetune_cfg.data.dataset_cfg.get('padd2same_size') if finetune_cfg.data.dataset_cfg.get('padd2same_size') else finetune_cfg.data.dataset_cfg.get('croppadd2same_size')

# %%
via11DS = CS_Dataset('bvisa', 'skull_stripped',
                     'central_sulcus',
                     dataset_path='/mrhome/vladyslavz/git/central-sulcus-analysis/data/brainvisa',
                     split='validation',
                     preload=False,
                     resample=finetune_cfg.data.dataset_cfg.resample,
                     croppadd2same_size=croppadd2same_size)

# %%
experiment_results = []
for idx in tqdm(range(len(via11DS))):
    sample = via11DS[idx]
    target_1hot = torch.nn.functional.one_hot(sample['target'].unsqueeze(0), num_classes=2).permute(0, 4, 1, 2, 3)

    with torch.no_grad():
        out = segm_model.forward(sample['image'].unsqueeze(0).to('cuda'))
    out_bin = torch.argmax(torch.softmax(out, dim=1), dim=1).cpu()
    out_1hot = torch.nn.functional.one_hot(out_bin, num_classes=2).permute(0, 4, 1, 2, 3)

    dice = compute_dice(out_1hot, target_1hot, include_background=False)
    iou = compute_iou(out_1hot, target_1hot, include_background=False)
    hausdorff_distance = compute_hausdorff_distance(out_1hot, target_1hot,
                                                    include_background=False)

    res = {'caseid': via11DS.caseids[idx],
           'dice': dice.item(),
           'iou': iou.item(),
           'hausdorff_distance': hausdorff_distance.item()
           }
    experiment_results.append(res)
experiment_results = pd.DataFrame(experiment_results)
experiment_results = experiment_results.set_index('caseid')
experiment_results.loc['MEAN'] = experiment_results.mean()
experiment_results.loc['STD'] = experiment_results.std()
experiment_results.to_csv(f'{out_path}/bvisa_metrics.csv')


# %%
# plt.imshow(out_hot[0, 1, 120, :, :], cmap='gray', alpha=0.5)
# plt.imshow(target_1hot[0, 1, 120, :, :], cmap='gray', alpha=0.5)
# plt.show()



# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
# slice = 120
# axs.imshow(sample['image'][0, slice, :, :], cmap='gray', alpha=0.5)
# axs.imshow(sample['target'][slice, :, :], cmap='gray', alpha=0.5)
# plt.show()

# %%
pprint(experiment_results)

# %%



