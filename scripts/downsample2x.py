# %%
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import numpy as np
# %%
## tqdm for loading bars
from tqdm import tqdm

# import torch
## Torchvision
# import torchvision
# from torchvision import transforms
import SimpleITK as sitk
from multiprocessing import Pool

from src.utils.general import resample_volume, sitk_cropp_padd_img_to_size, crop_image_to_content

# %%
# from src.models.simclr import SimCLR as SimCLRD
# from src.data.self_supervised_dm import ContrastiveDataSet as ContrastiveDataSetD, ContrastiveDataModule

# %%
orig_path = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/data/via11/nobackup/synth_generated')
down_path = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/data/via11/nobackup/synth_generated1114x')

# %%
imgs = [x for x in orig_path.glob('*/image_t1_*.nii.gz')]

# %%
print('Total images to process: ', len(imgs))

# %%
def process_img(p):
    new_p = str(p).replace('synth_generated', 'synth_generated1114x')
    Path(new_p).parent.mkdir(parents=True, exist_ok=True)
    
    img = sitk.ReadImage(str(p))
    labmap = sitk.ReadImage(str(p).replace('image', 'labels'))
    
    res_img = resample_volume(img, [1.1, 1.1, 1.5], sitk.sitkLinear)
    res_labmap = resample_volume(labmap, [1.1, 1.1, 1.5], sitk.sitkNearestNeighbor)

    res_img_array = sitk.GetArrayFromImage(res_img)
    res_labmap_array = sitk.GetArrayFromImage(res_labmap)
    
    res_labmap_array = ((res_labmap_array == 3)|(res_labmap_array == 42)).astype(np.uint8)
    
    res_img_cropped, min_coords, max_coord = crop_image_to_content(res_img_array)
    res_labmap_croped, _, __ = crop_image_to_content(res_labmap_array, min_coords, max_coord)

    res_img = sitk_cropp_padd_img_to_size(sitk.GetImageFromArray(res_img_cropped), [256, 256, 124])
    res_labmap = sitk_cropp_padd_img_to_size(sitk.GetImageFromArray(res_labmap_croped), [256, 256, 124])

    sitk.WriteImage(res_img, new_p)
    sitk.WriteImage(res_labmap, str(new_p).replace('image', 'labels'))
    return 1

# %% [markdown]
# Use multiprocessing

# %%
with Pool(10) as p:
    r = list(tqdm(p.imap(process_img, imgs[:]), total=len(imgs)))
