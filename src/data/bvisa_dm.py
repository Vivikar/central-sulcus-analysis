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


class CS_Dataset(Dataset):
    def __init__(self,
                 dataset: str,
                 split: str,
                 target: str,
                 dataset_path: str):
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
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(self.img_paths[idx])))
        target = None

        caseid = self.img_paths[idx].parent.parent.parent.name

        if self.target == 'sulci':
            lsulci = sitk.ReadImage(str(self.target_paths[idx][0]))
            rsulci = sitk.ReadImage(str(self.target_paths[idx][1]))
            target = lsulci + rsulci
            target = sitk.GetArrayFromImage(target)

        target = ((target == 48) | (target == 70)).astype(np.uint8)

        # add signle channel dimension
        image = np.expand_dims(image, axis=0)
        target = np.expand_dims(target, axis=0)
        image = torch.Tensor(image)
        target = torch.tensor(target, dtype=torch.long)
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
