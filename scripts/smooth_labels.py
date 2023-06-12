from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk
from src.utils.general import gaussian_distance_map
from src.data.bvisa_augm_dm import CS_Dataset

PATH2PROCESS = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/data/brainvisa_augm/nobackup/generated')

files2process = list(PATH2PROCESS.glob('**/labels*.nii.gz'))

def process_img(img_path):
    img = sitk.ReadImage(str(img_path))
    img_array = sitk.GetArrayFromImage(img)
    ls, rs = CS_Dataset.remove_btissue_labels(img_array)

    # extract CS segmentations
    target = ((ls == 48) | (rs == 70)).astype(np.float16)
    target = gaussian_distance_map(target, 2, 5)
    target = sitk.GetImageFromArray(target)
    target.CopyInformation(img)

    output_path = str(img_path).replace('labels', 'CSsmoothed_labels')
    sitk.WriteImage(target, output_path)


with Pool(10) as p:
    r = list(tqdm(p.imap(process_img, files2process),
                  total=len(files2process)))
