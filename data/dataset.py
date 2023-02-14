from pathlib import Path

import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from torchvision import datasets

from data.splits import bvisa_splits


class CS_Dataset(Dataset):
    def __init__(self,
                 dataset: str,
                 split: str,
                 target: str='sulci',
                 dataset_path: str|None=None):
        """Constructor for CS_Dataset class

        Args:
            dataset (str): Dataset name. Either 'bvisa' or .
            split (str): 'train', 'validation' or 'test' based on the splits.py.
            target (str, optional): Which image/object to load as target. Defaults to 'sulci'.
            dataset_path (str | None, optional): Path to the dataset folder with the
                directories of subjects in BrainVisa format. Defaults to None.
        """
        
        # save dataset hyperparameters
        self.target = target
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.split = split
        
        # load corresponding image and target paths
        self.img_paths = []
        self.target_paths = []
        if dataset == 'bvisa':
            self._load_bvisa(dataset_path)
        else:
            raise ValueError(f'Dataset: {dataset} not Implemented')
    
    def _load_bvisa(self, path: str|None):
        if path is None:
            path = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/data/brainvisa')
        else:
            path = Path(path)
                
        for subj in path.iterdir():
            if subj.is_dir() and subj.name in bvisa_splits[self.split]:
                self.img_paths.append(subj/f't1mri/t1/{subj.name}.nii.gz')
                
                if self.target == 'sulci':
                    lsulci = subj/f't1mri/t1/default_analysis/folds/3.3/base2018_manual/segmentation/LSulci_{subj.name}_base2018_manual.nii.gz'
                    rsulci = subj/f't1mri/t1/default_analysis/folds/3.3/base2018_manual/segmentation/RSulci_{subj.name}_base2018_manual.nii.gz'
                    self.target_paths.append((lsulci, rsulci))
                else:
                    raise ValueError(f'Target: {self.target} not Implemented')
    
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(self.img_paths[idx])))
        target = None
        
        caseid = self.img_paths[idx].parent.parent.parent.name
        
        if self.target == 'sulci':
            lsulci = sitk.ReadImage(str(self.target_paths[idx][0]))
            rsulci = sitk.ReadImage(str(self.target_paths[idx][1]))
            target = lsulci + rsulci
            target = sitk.GetArrayFromImage(target)
        
        return {'image': image, 'target': target, 'caseid': caseid}
        
