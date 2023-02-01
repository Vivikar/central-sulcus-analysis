from pathlib import Path
import SimpleITK as sitk
import logging
import numpy as np
import sys
import open3d as o3d

# Mesh properties to analyze
# ['cluster_properties',
#  'mesh_properties',
# 'holes_filled_volume',
# ]

# TODO: Add spike Hausdorf distance measure

ALL_PROPERTIES = ['cluster_properties',
                  'mesh_properties',
                  'holes_filled_volume']
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
    
    