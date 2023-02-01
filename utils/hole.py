from hole_filling_liepa.core import fill_hole_liepa, find_boundary_loops
from hole_filling_liepa.utils import read_obj, write_obj
import open3d as o3d
import numpy as np
import igl

mesh = o3d.io.read_triangle_mesh("/mnt/projects/VIA_Vlad/nobackup/BrainVisa/BrainVisa/CFIN/sub-via248/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_best/segmentation/RSulci_sub-via248_default_session_best.ply")
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

# vertices = np.asarray(mesh.vertices)
# faces = np.asarray(mesh.triangles)


vertices, faces = igl.read_triangle_mesh("/mnt/projects/VIA_Vlad/nobackup/BrainVisa/BrainVisa/CFIN/sub-via248/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_best/segmentation/RSulci_sub-via248_default_session_best.ply")

# vertices, faces = read_obj('mesh.obj')
boundary_loops = find_boundary_loops(faces)
print(boundary_loops)
patch_faces = fill_hole_liepa(vertices, faces, boundary_loops[0], method='angle')
# write_obj('patch.obj', vertices, patch_faces)