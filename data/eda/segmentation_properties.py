from pathlib import Path
import SimpleITK as sitk
import logging
import numpy as np
import sys
import open3d as o3d
from data.config import SAMSEG_PATH
import nibabel as nb

# Mesh properties to analyze
# ['cluster_properties',
#  'mesh_properties',
# 'holes_filled_volume',
# ]


ALL_PROPERTIES = ['cs_segmentation_overlap']
class SegmAnalyzer:
    def __init__(self, properties:list[str]=ALL_PROPERTIES) -> None:
        self.properties = properties
    
    def process(self, cs_segm, caseid) -> dict[str, float]:
        """_summary_

        Args:
            segm (np.ndarray): Should contain 0 for background, 1 for the Left CS
                and 2 for the Right CS.
            

        Returns:
            dict[str, float]: _description_
        """
        # load segmentations
        caseid = caseid.split('-')[-1]
        segm_path = SAMSEG_PATH + f'/{caseid}/mri/samseg/seg.mgz'
        samseg_segm = nb.load(segm_path).get_fdata().T
        features = {}
        
        for feat in self.properties:
            features = features | getattr(self, feat)(cs_segm, samseg_segm)
        return features
    
    @staticmethod
    def cs_segmentation_overlap(cs_segm, samseg_segm) -> dict:
        
        bvisa = sitk.GetArrayFromImage(cs_segm)
        
        left_cs = (bvisa == 1)
        right_cs = (bvisa == 2)
        
        lstats = {int(k):v for k,v in zip(*np.unique(samseg_segm[left_cs], return_counts=True))}
        rstats = {int(k):v for k,v in zip(*np.unique(samseg_segm[right_cs], return_counts=True))}
        
        return {'LCS_lebel2inters':lstats,
                'RCS_lebel2inters':rstats}
    
    