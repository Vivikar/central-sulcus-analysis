from pathlib import Path
import logging

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core import LightningDataModule
import os
from src.data.splits import bvisa_splits

logger = logging.getLogger(__name__)

lab_map = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 26: 24, 27: 25, 28: 26, 29: 27, 30: 28, 31: 29, 32: 30, 33: 31, 34: 32, 35: 33, 36: 34, 37: 35, 38: 36, 39: 37, 40: 38, 41: 39, 42: 40, 43: 41, 44: 42, 45: 43, 46: 44, 47: 45, 48: 46, 49: 47, 50: 48, 51: 49, 52: 50, 53: 51, 54: 52, 55: 53, 56: 54, 57: 55, 58: 56, 59: 57, 60: 58, 61: 59, 62: 60, 63: 61, 64: 62}

class CS_Dataset(Dataset):
    def __init__(self,
                 dataset: str,
                 split: str,
                 images: str,
                 target: str,
                 dataset_path: str):
        """Constructor for CS_Dataset class

        Args:
            dataset (str): Dataset name. Either 'bvisa' or .
            split (str): 'train', 'validation' or 'test' based on the splits.py.
            data (str): 'folds', 'skull_stripped' to train on fold skeletons or skull-stripped images.
            target (str, optional): Which image/object to load as target. Defaults to 'sulci'.
            dataset_path (str | None, optional): Path to the dataset folder with the
                directories of subjects in BrainVisa format. Defaults to None.
        """

        # save dataset hyperparameters
        self.target = target
        self.images = images
        self.dataset = dataset
        self.split = split
        self.dataset_path = Path(dataset_path)

        # load corresponding image and target paths
        self.img_paths = []
        self.target_paths = []

        if self.dataset == 'bvisa':
            self._load_bvisa()
        else:
            raise ValueError(f'Dataset: {dataset} not Implemented')

    def _load_bvisa(self):
        sulci_path = os.environ.get('SULCI_PATH_PREFIX')

        for subj in self.dataset_path.iterdir():
            if subj.is_dir() and subj.name in bvisa_splits[self.split]:
                self.img_paths.append(subj/f't1mri/t1/{subj.name}.nii.gz')

                if self.target == 'sulci':
                    lsulci = subj/f'{sulci_path}/LSulci_{subj.name}_base2018_manual.nii.gz'
                    rsulci = subj/f'{sulci_path}/RSulci_{subj.name}_base2018_manual.nii.gz'
                    self.target_paths.append((lsulci, rsulci))
                else:
                    raise ValueError(f'Target: {self.target} not Implemented')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.images == 'skull_stripped':
            image = sitk.GetArrayFromImage(sitk.ReadImage(str(self.img_paths[idx])))
        elif self.images == 'folds':
            folds_path = str(self.img_paths[idx])
            lfold = folds_path.replace('t1mri/t1/', 't1mri/t1/default_analysis/segmentation/Lskeleton_')
            rfold = folds_path.replace('t1mri/t1/', 't1mri/t1/default_analysis/segmentation/Rskeleton_')

            image = sitk.GetArrayFromImage(sitk.ReadImage(lfold)>11) #+\
                    # sitk.GetArrayFromImage(sitk.ReadImage(rfold)>11)
            image = (image>0).astype(np.int16)

        else:
            raise ValueError(f'Images: {self.images} not Implemented')
        target = None

        caseid = self.img_paths[idx].parent.parent.parent.name

        if self.target == 'sulci':
            lsulci = sitk.ReadImage(str(self.target_paths[idx][0]))
            # rsulci = sitk.ReadImage(str(self.target_paths[idx][1]))
            target = lsulci #+ rsulci
            target = sitk.GetArrayFromImage(target)

        # target = ((target == 48) | (target == 70)).astype(np.int16)
        target_remapped = np.zeros_like(target)
        for lab, new_lab in lab_map.items():
            target_remapped[target == lab] = new_lab
        
        # add signle channel dimension
        image = np.expand_dims(image, axis=0)
        # target = np.expand_dims(target, axis=0)
        image = torch.Tensor(image)
        target = torch.tensor(target_remapped, dtype=torch.long)
        return {'image': image, 'target': target, 'caseid': caseid}


class CS_DataModule(LightningDataModule):
    def __init__(self,
                 dataset_cfg: dict,
                 train_batch_size: int = 1,
                 validation_batch_size: int = 1,
                 num_workers: int = 1,

                 ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.num_workers = num_workers

        self.train_dataset = CS_Dataset(split='train', **dataset_cfg)

        self.val_dataset = CS_Dataset(split='validation', **dataset_cfg)

        logger.info(f'Len of train examples {len(self.train_dataset)} ' +
                    f'len of validation examples {len(self.val_dataset)}')

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers)

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.validation_batch_size,
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader
