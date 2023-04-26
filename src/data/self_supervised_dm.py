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
from src.data.splits import synthseg_sst_splits, bvisa_splits, via11_splits
from src.utils.general import sitk_cropp_padd_img_to_size, crop_image_to_content

logger = logging.getLogger(__name__)

# path to synstheg generated images and label maps folders
sytnthseg_orig_path = Path(os.environ['SYNTHSEG_PATH'])
sytnthseg2x2x2_path = Path(os.environ['SYNTHSEG2x2x2_PATH'])


class ContrastiveDataSet(data.Dataset):
    def __init__(self,
                 dataset: str,
                 split: str,
                 nviews: int = 2,
                 use_2x2x2_preproc: bool = True,
                 skull_strip: bool | float | str = True,
                 resample: list[float] | None = None,
                 croppadd2samesize: bool | str = False,
                 crop2content: bool = False,
                 ):
        """ContrastiveDataSet

        Args:
            dataset (str): Which dataset to use for contrastive learning.
                Available datasets: 'synthseg'.
            nviews (int, optional): How many agumented images of the same labelmap tp load . Defaults to 2.
            split (srt): Train or validation split.
            skull_strip (bool|float|str, optional): Whether to load skull-stripped images or not. 
                If a float given, removes the skull with a given probability.
                Defaults to True (always removes the skull & face).
            resample (list[x, y, z] | None, optional): Resample the images to a given resolution.

        """
        self.dataset = dataset
        self.nviews = nviews
        self.crop2content = crop2content
        self.skull_strip = skull_strip
        self.resample = list(resample) if resample is not None else None
        self.split = split
        self.use_2x2x2_preproc = use_2x2x2_preproc
        self.croppadd2samesize = croppadd2samesize
        self.transfrom = Affine(scale_params=(1.5, 1.5, 1.5))
        if dataset == 'synthseg':
            self._load_synthseg()
        elif dataset == 'brainvisa':
            self._load_brainvisa()
        elif dataset == 'via11':
            self._loadvia11()
        else:
            raise ValueError(f'Unknown dataset: {dataset}')

    def __getitem__(self, index):
        img_path = self.img_dirs[index]
        all_img_views = [x for x in img_path.glob('image*.nii.gz')]
        views_paths = [all_img_views[random.randint(0,len(all_img_views)-1)]\
                        for i in range(self.nviews)]

        views_images = self._load_images(views_paths)

        return (views_images, -1)

    def get_N_samples(self, index, N_samples):
        img_path = self.img_dirs[index]
        all_img_views = [x for x in img_path.glob('image*.nii.gz')]
        views_paths = all_img_views[:N_samples]

        views_images = self._load_images(views_paths)
        # views_paths_targets = [str(x).replace('image', 'labels') for x in views_paths]
        # views_targets = self._load_targets(views_paths_targets)
        return views_images

    def __len__(self):
        return len(self.img_dirs)

    def _loadvia11(self):
        if self.use_2x2x2_preproc:
            bvisa_path = Path(os.environ['VIA11_AUGM_PATH_2x'])
        else:
            bvisa_path = Path(os.environ['VIA11_AUGM_PATH'])

        self.img_dirs = [x for x in bvisa_path.iterdir() if x.is_dir()]

        if self.split == 'train':
            self.img_dirs = [x for x in self.img_dirs if x.name in via11_splits['train']]
        elif self.split == 'val':
            self.img_dirs = [x for x in self.img_dirs if x.name in via11_splits['validation']]

    def _load_brainvisa(self):
        if self.use_2x2x2_preproc:
            bvisa_path = Path(os.environ['BRAIN_VISA_AUGM_PATH_2x'])
        else:
            bvisa_path = Path(os.environ['BRAIN_VISA_AUGM_PATH'])

        self.img_dirs = [x for x in bvisa_path.iterdir() if x.is_dir()]

        if self.split == 'train':
            self.img_dirs = [x for x in self.img_dirs if x.name not in bvisa_splits['validation'] + bvisa_splits['test']]
        elif self.split == 'val':
            self.img_dirs = [x for x in self.img_dirs if x.name in bvisa_splits['validation']]

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

        # load and convert to numpy
        images = [sitk.ReadImage(str(p)) for p in views_paths]

        if self.dataset == 'synthseg':
            # reorient images
            images = [sitk.DICOMOrient(i, 'LAS') for i in images]
        elif self.dataset == 'via11':
            images = [sitk.DICOMOrient(i, 'LAS') for i in images]

        images = [self._preporces_sitk(img, views_paths[idx]) for idx, img in enumerate(images)]
        images = [sitk.GetArrayFromImage(img) for img in images]

        # skull strip
        if self.skull_strip:
            if not isinstance(self.skull_strip, str) and self.skull_strip:
                images = self._skull_strip(images, views_paths)
            elif self.skull_strip == 'half' and self.dataset != 'brainvisa':
                # remove skull from half of the images per each pair
                skull_stripped_imgs = self._skull_strip(images, views_paths)
                half_len = len(images)//2
                images = images[:half_len] + skull_stripped_imgs[half_len:]
                # crop to content

        # min-max normalization
        images = [(i - i.min())/(i.max() - i.min()) for i in images]

        # convert to torch tensors and return
        images = [torch.tensor(i, dtype=torch.float32) for i in images]

        # squeeze the channel dimension
        images = [i.unsqueeze(0) for i in images]
        # transformed = self.transfrom(images[0])[0].as_tensor()
        # images[1] = transformed
        return images

    def _skull_strip(self, images, image_paths):

        # if skull_strip is a float, remove the skull with a given probability
        if isinstance(self.skull_strip, float) and 0.<= self.skull_strip <= 1.:
            if random.random() >= self.skull_strip:
                return images

        # load labels
        lable_paths = [str(x).replace('image', 'labels') for x in image_paths]
        labels = [sitk.ReadImage(p) for p in lable_paths]
        if self.dataset == 'synthseg':
            # reorient images
            labels = [sitk.DICOMOrient(i, 'LAS') for i in labels]
        labels = [self._preporces_sitk(img, image_paths[idx], labelmap=True) for idx, img in enumerate(labels)]

        labels = [sitk.GetArrayFromImage(img) for img in labels]

        # mask outeverything except cortex labels
        cortex_mask = [((x < 500) & (x != 0)).astype(np.int16) for x in labels]
        masked_images = [x*y for x, y in zip(images, cortex_mask)]
        return masked_images

    def _preporces_sitk(self, img, path, labelmap=False):
        image_interpolator = sitk.sitkLinear if not labelmap else sitk.sitkNearestNeighbor
        if self.resample is not None:
            img = resample_volume(img, self.resample, image_interpolator)

        if self.crop2content:
            img_array = sitk.GetArrayFromImage(img)

            # load and use the mask to zero-out non-brain voxels
            mask = sitk.ReadImage(str(path).replace('image', 'labels'))
            if self.dataset == 'synthseg':
                mask = sitk.DICOMOrient(mask, 'LAS')
            elif self.dataset == 'via11':
                mask = sitk.DICOMOrient(mask, 'LAS')
            if self.resample:
                mask = resample_volume(mask, self.resample, sitk.sitkNearestNeighbor)
            mask = sitk.GetArrayFromImage(mask)
            img_array[mask == 0] = 0

            cropped_img = crop_image_to_content(img_array)[0]
            cropped_img = sitk.GetImageFromArray(cropped_img)
            cropped_img.SetDirection(img.GetDirection())
            cropped_img.SetOrigin(img.GetOrigin())
            cropped_img.SetSpacing(img.GetSpacing())
            img = cropped_img

        if self.croppadd2samesize:
            croppadd = [int(x) for x in self.croppadd2samesize.split('-')]
            img = sitk_cropp_padd_img_to_size(img, croppadd, 0)
        
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
