import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

from data.image import CS_Images
from data.config import BRAIN_VISA_PATH, CS_CORRECTED
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import SimpleITK as sitk

from skimage import measure

cs_paths = [x for x in Path(BRAIN_VISA_PATH).glob(CS_CORRECTED)]
dataset = CS_Images(segmentation='all',
                    mesh=True,
                    preload=False)

s0 = dataset[2]
img = sitk.GetArrayFromImage(s0['img'])
bvisa = sitk.GetArrayFromImage(s0['bvisa'])
corrected = sitk.GetArrayFromImage(s0['corrected'])
caseid = s0['caseid']
center = s0['centre']
slc = 100

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

axs[0].imshow(img[slc, :, :].T, cmap='gray')
axs[0].imshow(bvisa[slc, :, :].T, alpha=0.5, cmap='gray')
axs[0].set_axis_off()

axs[1].imshow(img[slc, :, :].T, cmap='gray')
axs[1].imshow(corrected[slc, :, :].T, alpha=0.5, cmap='gray')
axs[1].set_axis_off()

plt.tight_layout()
plt.show()

from skimage import measure, morphology

bvisa_cc = measure.label(bvisa, connectivity=2)

bvisa_dil = morphology.closing(bvisa, morphology.ball(3))
bvisa_dil = morphology.closing(bvisa_dil, morphology.ball(3))
bvisa_dil_cc = measure.label(bvisa_dil, connectivity=2)

print(np.unique(bvisa_cc, return_counts=True))
print(np.unique(bvisa_dil_cc, return_counts=True))

SE_f = np.zeros((5, 5, 5), dtype=np.bool_)
SE_f[1, 0, 2] = True #z (green), y (red), x (blue)
SE_f[1, 4, 2] = True
SE_f[2, 0, 2] = True
SE_f[2, 4, 2] = True
SE_f[3, 0, 2] = True
SE_f[3, 4, 2] = True
# print(SE_f)

SE_b = np.zeros((5, 5, 5), dtype=np.bool_)
SE_b[1, 2, 2] = True

SE_b[2, 2, 2] = True
SE_b[2, 2, 3] = True
SE_b[2, 3, 2] = True
SE_b[2, 1, 2] = True
SE_b[2, 2, 1] = True

SE_b[3, 2, 2] = True

# print('\n______________\n', SE_b)

# SE_rotX = [np.rot90(SE_f, k=1, axes=(0, 1)),]

SE_bank = [(SE_f, SE_b),
           # 90 deg rotation around x axis
           (np.rot90(SE_f, k=1, axes=(0, 1)),
            np.rot90(SE_b, k=1, axes=(0, 1))),
           # 90 deg rotation around y axis
           (np.rot90(SE_f, k=1, axes=(0, 2)),
            np.rot90(SE_b, k=1, axes=(0, 2)))]

np.unique(bvisa, return_counts=True)

bvisa_bin = bvisa == 1
hit_miss_result = np.zeros_like(bvisa, dtype=np.bool_)
for SE_f, SE_b in SE_bank:
    hit = morphology.erosion(bvisa_bin, SE_f)
    miss = morphology.erosion(~bvisa_bin, SE_b)
    hit_miss = hit & miss

    hit_miss_result = hit_miss_result | hit_miss