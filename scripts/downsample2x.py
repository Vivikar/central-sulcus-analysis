# %%
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))


# %%
## tqdm for loading bars
from tqdm import tqdm

# import torch
## Torchvision
# import torchvision
# from torchvision import transforms
import SimpleITK as sitk
from multiprocessing import Pool

from src.utils.general import resample_volume

# %%
# from src.models.simclr import SimCLR as SimCLRD
# from src.data.self_supervised_dm import ContrastiveDataSet as ContrastiveDataSetD, ContrastiveDataModule

# %%
orig_path = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/data/via11/nobackup/synth_generated')
down_path = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/data/via11/nobackup/synth_generated2x')

# %%
imgs = [x for x in orig_path.glob('*/*_t1_*.nii.gz')]

# %%
len(imgs)

# %%
def process_img(p):
    new_p = str(p).replace('synth_generated', 'synth_generated2x')
    Path(new_p).parent.mkdir(parents=True, exist_ok=True)
    print(p)
    print(new_p)
    # img = sitk.ReadImage(str(p))

    image_interpolator = sitk.sitkNearestNeighbor if 'labels' in new_p else sitk.sitkLinear 
    print(image_interpolator)
    # res_img = resample_volume(img, [2, 2, 2], image_interpolator)
    # sitk.WriteImage(res_img, new_p)
    return 1

# %% [markdown]
# Use multiprocessing

# %%
with Pool(10) as p:
    r = list(tqdm(p.imap(process_img, imgs[:10]), total=len(imgs)))

# %%
