# %%
from sklearn.manifold import Isomap
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

from tqdm import tqdm

from pathlib import Path

import argparse

data_path = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/shape_features/data/nobackup')

# %%
print(f'Loading sulci distance matrix from {data_path.parent.parent.parent}')
sulci_distance_matrix = np.load(data_path.parent.parent.parent/'sulci_distance_matrix_corr.npy')
sulci_reg_keys = np.load(data_path.parent.parent.parent/'sulci_reg_keys_corr.npy')
print(f'Using sulci distance matrix {sulci_distance_matrix.shape} and keys {sulci_reg_keys.shape}')
# np.shape(sulci_distance_matrix)

# sulci_distance_matrix = sulci_distance_matrix + sulci_distance_matrix.T

# %% [markdown]
# # Choosing d-dimensionality for ISOMAP

# %% [markdown]
# Let Mdist be the input distance matrix. A distance matrix of D-dimensional Gaussian vectors was computed, with the same average square distance as Mdist, which we called Mrand. Using a loop, we computed the reconstruction error edist of the Isomap fitted on Mdist and the reconstruction error erand of the Isomap fitted on Mrand for the whole range of possible number of components for the Isomap, so for d∈[1, dim(Mdist)]. We then considered the number of components maximizing the ratio erand / edist as the intrinsic dimensionality of the manifold 

# %%
M_dist = sulci_distance_matrix
M_rand = np.random.rand(M_dist.shape[0], M_dist.shape[1])
M_rand = (M_rand/np.mean(M_rand))*np.mean(M_dist)

hp_resuluts = []

for nn in tqdm(range(3, 100)):
    dists_proportions = []
    for d in range(2, M_rand.shape[0]-1):
        try:
            isomap = Isomap(n_components=d, n_neighbors=nn, n_jobs=-1)
            isomap.fit(M_dist)
            e_dist = isomap.reconstruction_error()

            isomap = Isomap(n_components=d, n_neighbors=nn, n_jobs=-1)
            isomap.fit(M_rand)
            e_rand = isomap.reconstruction_error()

            dists_proportions.append((e_dist, e_rand/e_dist))

        except:
            dists_proportions.append(dists_proportions[-1])
    hp_resuluts.append(dists_proportions)

# %%
pd.to_pickle(hp_resuluts, '/mrhome/vladyslavz/git/central-sulcus-analysis/shape_features/data/nobackup/hp_resuluts_error_ext_double_corr.pkl')

# %%



