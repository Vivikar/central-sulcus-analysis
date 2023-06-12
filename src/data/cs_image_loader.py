from pathlib import Path
from data.config import BRAIN_VISA_PATH, SEGM_PATH
import SimpleITK as sitk
import logging
import numpy as np
import sys
from open3d import io 

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class CS_Images:
    
    def __init__(self,
                 segmentation: str|None,
                 mesh:bool=False,
                 preload: bool=False) -> None:
        """Constructor for CS_Image class

        Args:
            segmentation (str): Defines whether to also load the segmentations
                together with images. Can be one of the following:
                    None: No segmentation will be loaded, only the images
                    'corrected': Load the post-processed after BrainVisa segmentations
                    'brainvisa': Load the raw BrainVisa segmentations
                    'all': Load both the raw and post-processed segmentations
            mesh: (bool, optional): Whether to load the mesh files. Defaults to False.
            preload (bool, optional): Whether to load all the images at once
                or load on the go. Defaults to False.
        """
        self.segmentation = segmentation        
        self.preload = preload
        self.mesh = mesh
        
        self.imgs = []
        self.bvisa_segms = []
        self.corrected_segms = []
        self.centers = []
        self.caseids = []
        
        self.bvisa_meshes = []
        self.corrected_meshes = []
        
        self._get_paths()
        if self.preload:
            self._preload_images()
    
    
    def _preload_images(self):
        raise NotImplementedError
    
    def _get_paths(self):
        subjects_paths = [x for x in Path(BRAIN_VISA_PATH).glob('*/sub-via*') if x.is_dir()]
        self.imgs = [x/f't1mri/default_acquisition/{x.name}.nii.gz' for x in subjects_paths]
        imgs_exist = np.asarray([x.exists() for x in self.imgs])
        
        lsulci_orig = [x/f'{SEGM_PATH}/LSulci_{x.name}_default_session_best.nii.gz' for x in subjects_paths ]
        rsulci_orig = [x/f'{SEGM_PATH}/RSulci_{x.name}_default_session_best.nii.gz' for x in subjects_paths ]
        self.bvisa_segms = list(zip(lsulci_orig, rsulci_orig))
        bvisa_exist = np.asarray([x[0].exists() and x[1].exists() for x in self.bvisa_segms])
        
        lsulci_cleaned = [x/f'{SEGM_PATH}/LSulci_{x.name}_default_session_best_cleaned.nii.gz' for x in subjects_paths ]
        rsulci_cleaned = [x/f'{SEGM_PATH}/RSulci_{x.name}_default_session_best_cleaned.nii.gz' for x in subjects_paths ]
        self.corrected_segms = list(zip(lsulci_cleaned, rsulci_cleaned))
        corrected_exist = np.asarray([x[0].exists() and x[1].exists() for x in self.corrected_segms])

        # count the number of subjects
        found_subjects = len(self.imgs)        
        print(f'Found {found_subjects} subjects and {sum(imgs_exist)} MPRAGE images')
        
        if self.segmentation == 'brainvisa':
            self.bvisa_segms = [x for xidx, x in enumerate(self.bvisa_segms) if bvisa_exist[xidx] and imgs_exist[xidx]]
            self.imgs = [x for xidx, x in enumerate(self.imgs) if imgs_exist[xidx] and bvisa_exist[xidx]]
            print(f'Found {sum(bvisa_exist)} BrainVisa segmentations from {found_subjects} subjects')
        
        elif self.segmentation == 'corrected':
            self.corrected_segms = [x for xidx, x in enumerate(self.corrected_segms) if corrected_exist[xidx] and imgs_exist[xidx]]
            self.imgs = [x for xidx, x in enumerate(self.imgs) if imgs_exist[xidx] and corrected_exist[xidx]]
            print(f'Found {sum(corrected_exist)} corrected segmentations from {found_subjects} subjects')
        elif self.segmentation == 'all':
            self.corrected_segms = [x for xidx, x in enumerate(self.corrected_segms) if corrected_exist[xidx] and imgs_exist[xidx] and bvisa_exist[xidx]]
            self.bvisa_segms = [x for xidx, x in enumerate(self.bvisa_segms) if bvisa_exist[xidx] and imgs_exist[xidx] and corrected_exist[xidx]]
            self.imgs = [x for xidx, x in enumerate(self.imgs) if imgs_exist[xidx] and bvisa_exist[xidx] and corrected_exist[xidx]]
            print(f'Found {len(self.imgs)} subjects with both BrainVisa and corrected from {found_subjects} subjects')
        elif self.segmentation != None:
            raise ValueError(f'Unknown segmentation type: {self.segmentation}. Should be None, "raw", "brainvisa", "cleaned" or "all"')
        
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        if not self.preload:
            
            img = sitk.ReadImage(str(self.imgs[idx]))
            
            bvisa = None
            corrected = None
            bvisa_mesh_lscl = None
            bvisa_mesh_rscl = None
            corrected_mesh_lscl = None
            corrected_mesh_rscl = None
            
            caseid = self.imgs[idx].name.split('.')[0]
            centre = self.imgs[idx].parent.parent.parent.parent.name
            
            if self.segmentation == 'brainvisa' or self.segmentation == 'all':
                lsulci = sitk.ReadImage(str(self.bvisa_segms[idx][0]))
                rsulci = sitk.ReadImage(str(self.bvisa_segms[idx][1]))*2
                bvisa = lsulci + rsulci
                    
                if self.mesh:
                    bvisa_mesh_lscl = io.read_triangle_mesh(str(self.bvisa_segms[idx][0]).replace('.nii.gz', '.ply'))
                    bvisa_mesh_rscl = io.read_triangle_mesh(str(self.bvisa_segms[idx][1]).replace('.nii.gz', '.ply'))
                    if bvisa_mesh_lscl is None or bvisa_mesh_rscl is None:
                        raise ValueError(f'Could not load brainvisa mesh for {caseid}')
            
            if self.segmentation == 'corrected' or self.segmentation == 'all':
                lsulci = sitk.ReadImage(str(self.corrected_segms[idx][0]))
                rsulci = sitk.ReadImage(str(self.corrected_segms[idx][1]))*2
                corrected = lsulci + rsulci
                if self.mesh:
                    corrected_mesh_lscl = io.read_triangle_mesh(str(self.corrected_segms[idx][0]).replace('.nii.gz', '.ply'))
                    corrected_mesh_rscl = io.read_triangle_mesh(str(self.corrected_segms[idx][1]).replace('.nii.gz', '.ply'))
                    if corrected_mesh_lscl is None or corrected_mesh_rscl is None:
                        raise ValueError(f'Could not load corrected mesh for {caseid}')
            
            return {'img': img,
                    'centre': centre,
                    'caseid': caseid,
                    'bvisa': bvisa,
                    'corrected': corrected,
                    'bvisa_mesh_lscl': bvisa_mesh_lscl,
                    'bvisa_mesh_rscl': bvisa_mesh_rscl,
                    'corrected_mesh_lscl': corrected_mesh_lscl,
                    'corrected_mesh_rscl': corrected_mesh_rscl}
        else:
            raise NotImplementedError
        
    def get_caseidx(self, casename:str):
        """_summary_

        Args:
            casename (str): sub-via228

        Returns:
            _type_: _description_
        """
        return np.where([casename in str(x) for x in self.imgs])