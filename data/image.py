from pathlib import Path
from data.config import BRAIN_VISA_PATH
import SimpleITK as sitk

class CS_Image:
    
    def __init__(self, cs_paths:list[str]|None,
                 preload: bool=False) -> None:
        """Constructor for CS_Image class

        Args:
            cs_paths (list[str] | None): List of paths to CS corrected images
            preload (bool, optional): Whether to load all the images at once
                or load on the go. Defaults to False.
        """
        
        self.cs_paths = cs_paths
        self.preload = preload
        
        self.imgs = []
        self.cs_segms = []
        self.centers = []
        self.caseids = []
        
        # initilize the list of paths
        # base on the corrected cs_paths
        if self.cs_paths is not None:
            self._init_cs_corrected()
    
    def _init_cs_corrected(self):
        if not self.preload:
            self.cs_segms = self.cs_paths
            for c in self.cs_segms:
                center = str(c).replace(BRAIN_VISA_PATH,
                                                      '').split('/')[0]
                caseid = str(c).replace(BRAIN_VISA_PATH,
                                                      '').split('/')[1]
                self.centers.append(center)
                self.caseids.append(caseid)
                
                self.imgs.append(Path(BRAIN_VISA_PATH)/f'{center}/{caseid}/t1mri/default_acquisition/{caseid}.nii.gz')
            
    def __len__(self):
        return len(self.cs_segms)
    
    def __getitem__(self, idx):
        if not self.preload:
            img = sitk.ReadImage(str(self.imgs[idx]))
            cs_mask = sitk.ReadImage(str(self.cs_segms[idx]))
        else:
            raise NotImplementedError
        return {'img': img,
                'cs_mask': cs_mask,
                'center': self.centers[idx],
                'caseid': self.caseids[idx],
                'img_path': self.imgs[idx],
                'cs_mask_path': self.cs_segms[idx]}