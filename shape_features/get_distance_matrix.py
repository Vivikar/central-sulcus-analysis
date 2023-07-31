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
from dataloaders import VIA11_Corrected_CS_Loader
from shape_utils import process_sulc_pair, sulcus_dist
# %%
cs_ds = VIA11_Corrected_CS_Loader(bv_good=True, corrected=True, all_bv=False, preload=True)

sulci_list = cs_ds.get_subjects()
# %%
sulci_distance_matrix = []
sulci_reg_keys = []


ss = 0
for s in tqdm(sulci_list):
    dists, reg_keys = sulcus_dist(s, sulci_list[ss:])
    dists = [0]*ss + dists
    reg_keys = ['']*ss + reg_keys
    sulci_distance_matrix.append(dists)
    sulci_reg_keys.append(reg_keys)
    ss += 1

sulci_distance_matrix = np.array(sulci_distance_matrix)
sulci_reg_keys = np.array(sulci_reg_keys)

np.save('./sulci_distance_matrix_corr.npy', sulci_distance_matrix)
np.save('./sulci_reg_keys_corr.npy', sulci_reg_keys)
