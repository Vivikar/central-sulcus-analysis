# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import pickle
import trimesh
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader

import SimpleITK as sitk

import open3d as o3d

# import tqdm

import multiprocessing as mp

# %%
class VIA11_Corrected_CS_Loader(Dataset):
    def __init__(self,
                 bv_good: bool = True,
                 corrected:bool = True,
                 all_bv: bool = False,
                 preload: bool = False) -> None:
        """Constructor for VIA11_Corrected_CS_Loader

        Args:
            bv_good (bool, optional): Flag to indicate wether to load BrainVISA segmentations that are good (type 1).
                Defaults to True.
            corrected (bool, optional): Flag to indicate wether to load the manually corrected CS segmentations.
                Overrides any BrainVISA segmentations loaded. Defaults to True.
            all_bv (bool, optional): Flag to indicate wether to load all BrainVISA segmentations no matter their QC status.
                Overrides all other flags. Defaults to False.
        """
        
        self.bv_good = bv_good
        self.corrected = corrected
        self.all_bv = all_bv
        self.preload = preload
        
        # store segmentations in format 
        # {subject_id: sub-viaxxx, type': type_of_segmentation},
        #  'lsegm': path_to_left_segmentation, 'rsegm': path_to_right_segmentation, 
        #  'lmesh': path_to_left_mesh, 'rmesh': path_to_right_mesh}
        #               '}
        self.cs_segmentations = []
        self._load_data()
        
        pass
    
    def __len__(self):
        return len(self.cs_segmentations)
    
    def __getitem__(self, idx):
        if self.preload:
            d = self.cs_segmentations[idx]
            return {'subject_id': d['subject_id'],
                    'type': d['type'],
                    'lsegm': sitk.ReadImage(d['lsegm']),
                    'rsegm': sitk.ReadImage(d['rsegm']),
                    'lmesh': o3d.io.read_triangle_mesh(d['lmesh']),
                    'rmesh': o3d.io.read_triangle_mesh(d['rmesh'])}
        else:
            return self.cs_segmentations[idx]
    
    def _load_data(self):
        corrected_loaded = []
        if self.corrected:
            corr_paths = Path('/mnt/projects/VIA_Vlad/nobackup/BrainVisa/CS_edited').glob('**/LSu*sub-via*.nii.gz')
            for lsegmpath in corr_paths:
                resgmpath = str(lsegmpath).replace('LSulci', 'RSulci')
                lmeshpath = str(lsegmpath).replace('.nii.gz', '.ply')
                rmeshpath = str(lmeshpath).replace('LSulci', 'RSulci')
                
                subj = lsegmpath.name[7:17]
                subj_data = {'subject_id': subj}
                subj_data['lsegm'] = lsegmpath
                subj_data['rsegm'] = resgmpath
                subj_data['lmesh'] = lmeshpath
                subj_data['rmesh'] = rmeshpath
                subj_data['type'] = 'corrected'
                subj_data['site'] = str(lsegmpath).split('/')[-2]
                self.cs_segmentations.append(subj_data)
                corrected_loaded.append(subj)
        
        if self.bv_good:
            # load QC results
            qc_results = pd.read_excel(Path('/mnt/projects/VIA_Vlad/nobackup/QA_centralSulcus_lkj.xlsx'))
            qc_results = qc_results.set_index('subjects')
            bv_good_subjs = qc_results[qc_results.vis_QA == 1]
            
            for subj, metadata in bv_good_subjs.iterrows():
                
                # skip BrainVISA segmentations if corrected segmentations are loaded
                if subj in corrected_loaded:
                    continue
                
                lsegmpath = f'/mnt/projects/VIA_Vlad/nobackup/BrainVisa/BrainVisa/{metadata.sites}/{subj}/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_best/segmentation/LSulci_{subj}_default_session_best.nii.gz'
                resgmpath = lsegmpath.replace('LSulci', 'RSulci')
                lmeshpath = lsegmpath.replace('.nii.gz', '.ply')
                rmeshpath = lmeshpath.replace('LSulci', 'RSulci')
                
                subj_data = {'subject_id': subj}
                subj_data['lsegm'] = lsegmpath
                subj_data['rsegm'] = resgmpath
                subj_data['lmesh'] = lmeshpath
                subj_data['rmesh'] = rmeshpath
                subj_data['type'] = 'bv_good'
                subj_data['site'] = metadata.sites
                self.cs_segmentations.append(subj_data)
        
        if self.all_bv:
            self.cs_segmentations = []
            # load QC results
            qc_results = pd.read_excel(Path('/mnt/projects/VIA_Vlad/nobackup/QA_centralSulcus_lkj.xlsx'))
            qc_results = qc_results.set_index('subjects')
            bv_good_subjs = qc_results[qc_results.vis_QA != 999]
            
            for subj, metadata in bv_good_subjs.iterrows():
                
                lsegmpath = f'/mnt/projects/VIA_Vlad/nobackup/BrainVisa/BrainVisa/{metadata.sites}/{subj}/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_best/segmentation/LSulci_{subj}_default_session_best.nii.gz'
                resgmpath = lsegmpath.replace('LSulci', 'RSulci')
                lmeshpath = lsegmpath.replace('.nii.gz', '.ply')
                rmeshpath = lmeshpath.replace('LSulci', 'RSulci')
                
                subj_data = {'subject_id': subj}
                subj_data['lsegm'] = lsegmpath
                subj_data['rsegm'] = resgmpath
                subj_data['lmesh'] = lmeshpath
                subj_data['rmesh'] = rmeshpath
                subj_data['type'] = 'bv_good'
                subj_data['site'] = metadata.sites
                self.cs_segmentations.append(subj_data)
        

# %%
ds = VIA11_Corrected_CS_Loader(bv_good=True, corrected=True, all_bv=False)
len(ds)
# Path(ds[0]['rmesh']).exists()

# %%
subject = ds[0]
print(subject)

lmesh = o3d.io.read_triangle_mesh(subject['lmesh'])
rmesh = o3d.io.read_triangle_mesh(subject['rmesh'])

lsegm = sitk.ReadImage(str(subject['lsegm']))
rsegm = sitk.ReadImage(str(subject['rsegm']))

# %% [markdown]
# 1. Mirroring

# %% [markdown]
# The right sulci were finally mirrored relative to the interhemispheric plane in order to facilitate the comparison with the left ones. https://www.sciencedirect.com/science/article/pii/S1053811921011083#sec0037

# %%
rsegm_flipped =  sitk.Flip(rsegm, [False, False, True])

# %% [markdown]
# 2. ICP Registstration

# %% [markdown]
# The dissimilarity between any two given sulci was then computed in the following way: sulcus A was registered to sulcus B using a rigid transformation, using the Iterative Closest Point algorithm (ICP) (Besl and McKay, 1992), and the residual distance dA→B between the two sulci after registration was captured using the Wasserstein distance (Dobrushin, 1970). 
# 
# https://www.sciencedirect.com/science/article/pii/S1053811921011083#sec0037

# %%
lsegm_points = np.stack(np.where(sitk.GetArrayFromImage(lsegm))).T
rsegm_points = np.stack(np.where(sitk.GetArrayFromImage(rsegm_flipped))).T

# %%
#Apply the iterative closest point algorithm to align a point cloud with another point cloud or mesh.
# Will only produce reasonable results if the initial transformation is roughly correct.
# Initial transformation can be found by applying Procrustes’ analysis to a suitable set of
# landmark points (often picked manually).

# finds the transformation matrix sending a to b
matrix, transformed, cost = trimesh.registration.icp(a=rsegm_points, b=lsegm_points, max_iterations=1000)


# %% [markdown]
# # ISOMAP

# %%
cs_ds = VIA11_Corrected_CS_Loader(bv_good=True, corrected=True, all_bv=False, preload=True)

sulci =[cs_ds[i] for i in tqdm(range(len(cs_ds)))]

# %%
# unravell all the sulci
sulci_list = []
for s in sulci:
    sulci_list.append(['left' ,s['subject_id'], s['type'], s['lsegm']])
    sulci_list.append(['right', s['subject_id'], s['type'], s['rsegm']])


def process_sulc_pair(x):
    sulc, sulc2 = x

# %%
# create a matrix of pairwise distances between sulci

def sulcus_dist(sulc: list, sulclist: list[list] = sulci_list):
    dists = []
    reg_keys = []
    for sulc2 in tqdm(sulclist):
        
        # register the sulci min(a->b or b->a)
        
        # need to flip if different hemispheres
        if sulc[0] != sulc2[0]:
            sulc2[3] = sitk.Flip(sulc2[3], [False, False, True])
            
        # register the sulci with ICP
        s1_points = np.stack(np.where(sitk.GetArrayFromImage(sulc[3]))).T
        s2_points = np.stack(np.where(sitk.GetArrayFromImage(sulc2[3]))).T
        
        # finds the transformation matrix sending a to b
        _, __, cost_s1_to_s2 = trimesh.registration.icp(a=s1_points, b=s2_points, max_iterations=1000)
        _, __, cost_s2_to_s1 = trimesh.registration.icp(a=s2_points, b=s1_points, max_iterations=1000)
        
        dists.append(min(cost_s1_to_s2, cost_s2_to_s1))
        
        reg_keys.append(f'{sulc[0]}_{sulc[1]}-{sulc2[0]}_{sulc2[1]}')

    return dists, reg_keys

# %%
sulci_distance_matrix = []
sulci_reg_keys = []

with mp.Pool(1) as p:
    r = list(tqdm(p.imap(sulcus_dist, sulci_list[:1]), total=1))
sulci_distance_matrix = [x[0] for x in r]
sulci_reg_keys = [x[1] for x in r]

sulci_distance_matrix = np.array(sulci_distance_matrix)
sulci_reg_keys = np.array(sulci_reg_keys)

np.save('./sulci_distance_matrix.npy', sulci_distance_matrix)
np.save('./sulci_reg_keys.npy', sulci_reg_keys)

# %%
sulci_distance_matrix = np.array(sulci_distance_matrix)
sulci_reg_keys = np.array(sulci_reg_keys)

np.save('./sulci_distance_matrix.npy', sulci_distance_matrix)
np.save('./sulci_reg_keys.npy', sulci_reg_keys)

# %%



