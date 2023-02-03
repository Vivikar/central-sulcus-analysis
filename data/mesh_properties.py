from pathlib import Path
import SimpleITK as sitk
import logging
import numpy as np
import sys
import open3d as o3d
from utils.mesh import dihedral_angle
# Mesh properties to analyze
# ['cluster_properties',
#  'mesh_properties',
# 'holes_filled_volume',
# ]

# TODO: Add spike Hausdorf distance measure

ALL_PROPERTIES = ['cluster_properties',
                  'mesh_properties',
                  'holes_filled_volume',
                  'dihedral_angles']
class MeshAnalyzer:
    def __init__(self, properties:list[str]=ALL_PROPERTIES) -> None:
        self.properties = properties
    
    def process(self, mesh:o3d.geometry.TriangleMesh) -> dict[str, float]:
        features = {}
        
        for feat in self.properties:
            features = features | getattr(self, feat)(mesh)
        return features
    
    @staticmethod
    def cluster_properties(mesh:o3d.geometry.TriangleMesh) -> dict:
        clust_idx, triangl_per_clust, srf_are_per_clust = mesh.cluster_connected_triangles()
        total_srf_area = np.sum(srf_are_per_clust)
        
        return {'cluster_connected_triangles': len(clust_idx),
                'triangl_per_clust': triangl_per_clust,
                'srf_are_per_clust':np.array(srf_are_per_clust),
                'total_srf_area': total_srf_area,
                }
    
    @staticmethod
    def mesh_properties(mesh:o3d.geometry.TriangleMesh) -> dict:
        """See http://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Visualize-a-3D-mesh
        for more details"""
        watertight = mesh.is_watertight()
        volume = -1
        if watertight:
            volume = mesh.get_volume()      

        return {'is_self_intersecting':mesh.is_self_intersecting(),
                'is_watertight':mesh.is_watertight(),
                'is_edge_manifold':mesh.is_edge_manifold(),
                'is_vertex_manifold' :mesh.is_vertex_manifold(),
                'total_volume':volume,

                }
    
    @staticmethod
    def holes_filled_volume(mesh:o3d.geometry.TriangleMesh) -> dict:
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        filled = mesh.fill_holes().to_legacy()
        if not filled.is_watertight():
            return {'holes_filled_volume':-1}
        return {'holes_filled_volume':filled.get_volume()}
    
    @staticmethod
    def hausdorff_distance(mesh1:o3d.geometry.TriangleMesh,
                           smooth_iter:3) -> dict:
        
        dist = 0
        return {'hausdorff_distance':dist}
    
    @staticmethod
    def dihedral_angles(mesh:o3d.geometry.TriangleMesh) -> dict:
        mesh.compute_triangle_normals()
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        triangle_normals = np.asarray(mesh.triangle_normals)
        
        edges = np.concatenate([triangles[:, [0, 1]],
                                triangles[:, [1, 2]],
                                triangles[:, [2, 0]]], axis=0)
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        
        edge_dihedral_angles = np.zeros((edges.shape[0]))
        
        for edgidx, edg in enumerate(edges):
            t1n = triangle_normals[edg[0]]
            t2n = triangle_normals[edg[1]]
            edge_dihedral_angles[edgidx] = dihedral_angle(t1n, t2n)
        
        return {'edge_dihedral_angles':edge_dihedral_angles}