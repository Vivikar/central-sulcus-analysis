import igl
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os

root_folder = os.getcwd()

v, f = igl.read_triangle_mesh("/mnt/projects/VIA_Vlad/nobackup/BrainVisa/BrainVisa/CFIN/sub-via227/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_best/segmentation/LSulci_sub-via227_default_session_best.ply")

U,UF,I = igl.project_isometrically_to_plane(v,f)

# plot the surface with matplotlib
plt.scatter(U[:,0],U[:,1],s=0.1)

plt.show()