from pathlib import Path
import pandas as pd
import numpy as np

from tqdm import tqdm
import subprocess 
import os


import SimpleITK as sitk


# import tqdm
import subprocess
from scipy.ndimage import gaussian_filter

from dataloaders import VIA11_Corrected_CS_Loader

from spam import SPAM


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

args = parser.parse_args()

if args.spam_sulci is None:
    if args.sulci_list is not None:
        print(f'Loading sulci list from file {args.sulci_list}')
        cs_ds, sulci_list = pd.read_pickle(data_path/args.sulci_list)
    else:
        cs_ds = VIA11_Corrected_CS_Loader(bv_good=True, corrected=True, all_bv=False, preload=True)
        # unravell all the sulci
        sulci_list = cs_ds.get_subjects()
        pd.to_pickle((cs_ds, sulci_list), data_path/'sulci_list.pkl')

    print(f'Using {cs_ds}', f'\nNumber of subjects: {len(cs_ds)}')


    print(f'Loading sulci distance matrix from {data_path.parent.parent.parent}')
    sulci_distance_matrix = np.load(data_path.parent.parent.parent/'sulci_distance_matrix.npy')
    sulci_reg_keys = np.load(data_path.parent.parent.parent/'sulci_reg_keys.npy')
    print(f'Using sulci distance matrix {sulci_distance_matrix.shape} and keys {sulci_reg_keys.shape}')

    if args.spam is not None:
        print(f'Loading SPAM from {args.spam}')
        spam = pd.read_pickle(data_path/args.spam)
    else:
        spam = SPAM(sulci_list, sulci_distance_matrix)
        pd.to_pickle(spam, data_path/'spam.pkl')
    print(f'Using SPAM {spam}')

    all_spam_sulci, isomap_feat_values = spam.retrieve_isomap_spams(sample_shapes_n=20,
                                            isomap_components=args.isomap_components,
                                            n_neighbors=args.n_neighbors,
                                            l=args.l,)
    pd.to_pickle((all_spam_sulci, isomap_feat_values), data_path/'spam_sulci.pkl')

else:
    print(f'Loading SPAM sulci from {data_path/args.spam_sulci}')
    all_spam_sulci, isomap_feat_values = pd.read_pickle(data_path/args.spam_sulci)


# %%
sigma = args.sigma
bin_thresh = args.bin_thresh
save_path = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/shape_features/meshes/test').absolute()

def save_mesh(segm_array:np.ndarray,
              sigma:float, bin_thresh:np.float32,
              label:str, save_path:Path,
              template_img:sitk.Image=None,):
    # smooth the segmentation
    segm_array = gaussian_filter(segm_array, sigma=sigma)

    # binarize the segmentation
    segm_array = (segm_array >= bin_thresh).astype(np.int16)

    segm_img = sitk.GetImageFromArray(segm_array)
    if template_img is not None:
        segm_img.CopyInformation(template_img)

    # save the segmentation
    sitk.WriteImage(segm_img, save_path/f'{label}.nii.gz')

    # transform it into a mesh
    cmd = ['/mrhome/vladyslavz/portable_apps/brainvisa4_5/bin/AimsMeshBrain', '-i', save_path/f'{label}.nii.gz', '-o', save_path/f'{label}.ply']
    subprocess.run(cmd, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.remove(save_path/f'{label}.nii.gz')
    os.remove(save_path/f'{label}.ply.minf')

# %%
orig_img = sitk.ReadImage('/mnt/projects/VIA_Vlad/nobackup/BrainVisa/CS_edited/CFIN/LSulci_sub-via010_default_session_best_edit_NHT.nii.gz')

for f in range(len(all_spam_sulci)):
    save_path_f = save_path/f'feature_{f}'
    save_path_f.mkdir(parents=True, exist_ok=True)
    # %%
    for i, s in tqdm(enumerate(all_spam_sulci[f]), total=len(all_spam_sulci[f])):
        save_mesh(s, sigma=sigma,
                bin_thresh=bin_thresh,
                label=f'test{i}', save_path=save_path_f, template_img=orig_img)
