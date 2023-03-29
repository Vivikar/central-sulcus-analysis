import os
import random
import logging
from pathlib import Path

import torch
import numpy as np
import SimpleITK as sitk
import torch.utils.data as data
from pytorch_lightning import LightningDataModule
from monai.transforms import Affine

from src.utils.general import resample_volume
from src.data.splits import synthseg_sst_splits

logger = logging.getLogger(__name__)

# path to synstheg generated images and label maps folders
sytnthseg_orig_path = Path(os.environ['SYNTHSEG_PATH'])
sytnthseg2x2x2_path = Path(os.environ['SYNTHSEG2x2x2_PATH'])


class ContrastiveDataSet(data.Dataset):
    def __init__(self,
                 dataset: str,
                 split: str,
                 nviews:int = 2,
                 use_2x2x2_preproc: bool = True,
                 skull_strip: bool | float = True,
                 resample: list[float] | None = None,
                 ):
        """ContrastiveDataSet

        Args:
            dataset (str): Which dataset to use for contrastive learning.
                Available datasets: 'synthseg'.
            nviews (int, optional): How many agumented images of the same labelmap tp load . Defaults to 2.
            split (srt): Train or validation split.
            skull_strip (bool|float, optional): Whether to load skull-stripped images or not. 
                If a float given, removes the skull with a given probability.
                Defaults to True (always removes the skull & face).
            resample (list[x, y, z] | None, optional): Resample the images to a given resolution.

        """
        self.dataset = dataset
        self.nviews = nviews
        self.skull_strip = skull_strip
        self.resample = list(resample) if resample is not None else None
        self.split = split
        self.use_2x2x2_preproc = use_2x2x2_preproc
        self.transfrom = Affine(scale_params=(1.5, 1.5, 1.5))
        if dataset == 'synthseg':
            self._load_synthseg()
        else:
            raise ValueError(f'Unknown dataset: {dataset}')

    def __getitem__(self, index):
        img_path = self.img_dirs[index]
        all_img_views = [x for x in img_path.glob('image*.nii.gz')]
        views_paths = [all_img_views[random.randint(0,len(all_img_views)-1)]\
                        for i in range(self.nviews)]

        views_images = self._load_images(views_paths)

        return views_images

    def __len__(self):
        return len(self.img_dirs)

    def _load_synthseg(self):
        if self.use_2x2x2_preproc:
            sytnthseg_path = sytnthseg2x2x2_path
        else:
            sytnthseg_path = sytnthseg_orig_path

        self.img_dirs = [x for x in sytnthseg_path.iterdir() if x.is_dir()]

        if self.split == 'train':
            self.img_dirs = [x for x in self.img_dirs if x.name not in synthseg_sst_splits['val']]
        elif self.split == 'val':
            self.img_dirs = [x for x in self.img_dirs if x.name in synthseg_sst_splits['val']]

    def _load_images(self, views_paths):

        # load images and convert to numpy
        images = [sitk.ReadImage(str(p)) for p in views_paths]
        images = [self._preporces_sitk(img) for img in images]
        images = [sitk.GetArrayFromImage(img) for img in images]

        # load labels and convert to numpy
        lable_paths = [str(x).replace('image', 'labels') for x in views_paths]
        labels = [sitk.ReadImage(p) for p in lable_paths]
        labels = [self._preporces_sitk(img, labelmap=True) for img in labels]
        labels = [sitk.GetArrayFromImage(img) for img in labels]

        # skull strip
        if self.skull_strip:
            # if skull_strip is a float, remove the skull with a given probability
            if random.random() < self.skull_strip:
                # mask out everything except cortex labels
                cortex_mask = [((x < 500) & (x != 0)).astype(np.int16) for x in labels]
                images = [x*y for x, y in zip(images, cortex_mask)]

        # min-max normalization
        images = [(i - i.min())/(i.max() - i.min()) for i in images]

        # convert to torch tensors and return
        images = [torch.tensor(i, dtype=torch.float32) for i in images]
        labels = [torch.tensor(i, dtype=torch.int64) for i in labels]

        # encode labels
        labels = [self._encode_segmentation(l) for l in labels]

        # squeeze the channel dimension
        images = [i.unsqueeze(0) for i in images]
        
        # CAREFULL REMOVE SIMPEL CASE TODO:::
        transformed = self.transfrom(images[0])[0].as_tensor()
        images[1] = transformed
        
        return images + labels

    def _encode_segmentation(self, img):
        # 0 - everything except, 1-GM (FreeSuerfer 3/42), 2-WM (FreeSurfer 2/41)
        img[(img != 3)&(img != 42)&(img != 2)&(img != 41)] = 0
        img[(img == 3)|(img == 42)] = 1
        img[(img == 2)|(img == 41)] = 2
        return img.unsqueeze(0)

    def _preporces_sitk(self, img, labelmap=False):
        image_interpolator = sitk.sitkLinear if not labelmap else sitk.sitkNearestNeighbor
        if self.resample is not None:
            img = resample_volume(img, self.resample, image_interpolator)
            return img
        else:
            return img


class ContrastiveDataModule(LightningDataModule):
    def __init__(self,
                 dataset_cfg,
                 train_batch_size: int,
                 validation_batch_size: int,
                 num_workers: int) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.num_workers = num_workers

        self.train_dataset = ContrastiveDataSet(**dataset_cfg, split='train')

        self.val_dataset = ContrastiveDataSet(**dataset_cfg, split='val')

        logger.info(f'Len of train examples {len(self.train_dataset)} ' +
                    f'len of validation examples {len(self.val_dataset)}')

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset,
                               batch_size=self.train_batch_size,
                               shuffle=True,
                               num_workers=self.num_workers,
                               drop_last=True,
                               pin_memory=True)

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.validation_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True)