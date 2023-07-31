from pathlib import Path
import pandas as pd
import numpy as np

from tqdm import tqdm
import subprocess 
import os
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import pandas as pd
import trimesh
import pyvista as pv
import matplotlib.pyplot as plt
import subprocess as sp
from dataloaders import VIA11_Corrected_CS_Loader

from PIL import Image, ImageDraw, ImageFont
from spam import SPAM
import SimpleITK as sitk

import os
# import tqdm
import subprocess
from scipy.ndimage import gaussian_filter

from dataloaders import VIA11_Corrected_CS_Loader


# create a 3d cross structuring element

se_3d_cross = np.array([[[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],
                       
                        [ [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 1, 1, 1, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0]],
                        
                        
                        [ [0, 0, 1, 0, 0],
                          [0, 1, 1, 1, 0],
                          [1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 0],
                          [0, 0, 1, 0, 0]],
                        
                        [ [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 1, 1, 1, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0]],
                        
                        [ [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]]])
                         

from spam import SPAM
from skimage.morphology import binary_dilation, binary_erosion, disk, binary_closing, ball

from pathlib import Path

import argparse

data_path = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/shape_features/data/nobackup')

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('--sulci_list', metavar='sulci_list', type=str, default=None)
parser.add_argument('--sulci_distance_matrix', metavar='sulci_distance_matrix',
                    type=str, default=None)
parser.add_argument('--spam', metavar='spam', type=str, default=None)
parser.add_argument('--spam_sulci', metavar='spam_sulci', type=str, default=None)
parser.add_argument('--sigma', metavar='sigma', type=float, default=0.8)
parser.add_argument('--bin_thresh', metavar='bin_thresh', type=float, default=0.2)
parser.add_argument('--isomap_components', metavar='isomap_components', type=int, default=10)
parser.add_argument('--n_neighbors', metavar='n_neighbors', type=int, default=10)
parser.add_argument('--sample_shapes_n', metavar='sample_shapes_n', type=int, default=10)
parser.add_argument('--l', metavar='l', type=float, default=20)
isomap_feat_values = None
spam = None
args = parser.parse_args()
spam_sulci = None
sulci_distance_matrix = None
if args.spam_sulci is None:
    if args.sulci_list is not None:
        print(f'Loading sulci list from file {args.sulci_list}')
        cs_ds, sulci_list = pd.read_pickle(data_path/args.sulci_list)
    else:
        cs_ds = VIA11_Corrected_CS_Loader(bv_good=True, corrected=True, all_bv=False, preload=True)
        # unravell all the sulci
        sulci_list = cs_ds.get_subjects()

    print(f'Using {cs_ds}', f'\nNumber of subjects: {len(cs_ds)}')


    print(f'Loading sulci distance matrix from {data_path.parent.parent.parent}')
    sulci_distance_matrix = np.load(data_path.parent.parent.parent/'sulci_distance_matrix.npy')
    sulci_reg_keys = np.load(data_path.parent.parent.parent/'sulci_reg_keys.npy')
    print(f'Using sulci distance matrix {sulci_distance_matrix.shape} and keys {sulci_reg_keys.shape}')

allsubj_meshp = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/shape_features/meshes/all_subj_meshes_reg')

spam = SPAM(sulci_list, sulci_distance_matrix)

all_spam_sulci, isomap_feat_values = spam.retrieve_isomap_spams(sample_shapes_n=20,
                                        isomap_components=args.isomap_components,
                                        n_neighbors=args.n_neighbors,
                                        l=args.l,)
i = sitk.ReadImage('/mnt/projects/VIA_Vlad/nobackup/BrainVisa/CS_edited/DRCMR/LSulci_sub-via004_default_session_best_edit_NHT.nii.gz')
# creating ply files
print('Creating ply files')
for subjidx in tqdm(range(len(spam.sulci_list))):
    subj  = spam.sulci_list[subjidx]
    # print(str(allsubj_meshp/f'{subj[1]}_{subj[0]}.nii.gz'))
    # ['left', s['subject_id'], s['type'], sitk.Image]
    
    # remove holes
    sub_img = sitk.GetArrayFromImage(subj[3])
    # sub_img = binary_dilation(sub_img, selem=ball(3)).astype(np.int16)
    sub_img = binary_closing(sub_img, se_3d_cross).astype(np.int16)
    sub_img = sitk.GetImageFromArray(sub_img)
    subj[3] = sub_img
    
    subj[3] = sitk.Cast(subj[3], sitk.sitkInt16)
    subj[3].CopyInformation(i)
    img_saved = sitk.WriteImage(subj[3],
                                str(allsubj_meshp/f'{subj[1]}_{subj[0]}.nii.gz'))
    # transform it into a mesh
    cmd = ['/mrhome/vladyslavz/portable_apps/brainvisa4_5/bin/AimsMeshBrain',
           '-i', str(allsubj_meshp/f'{subj[1]}_{subj[0]}.nii.gz'),
                     '-o', str(allsubj_meshp/f'{subj[1]}_{subj[0]}.ply')]
    subprocess.run(cmd, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(str(allsubj_meshp/f'{subj[1]}_{subj[0]}.ply'))
    # os.remove(str(allsubj_meshp/f'{subj[1]}_{subj[0]}.nii.gz'))
    # os.remove(str(allsubj_meshp/f'{subj[1]}_{subj[0]}.ply.minf'))