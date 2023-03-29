from pathlib import Path
import logging

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core import LightningDataModule
from monai.transforms import SpatialPad, Rotate # TODO: application of them causes mistakes
from torchvision.transforms import RandomRotation
import os
from src.data.splits import (bvisa_splits, bvisa_left_sulci_labels,
                             bvisa_right_sulci_labels, bvisa_padding_dims)
from src.utils.general import crop_image_to_content, resample_volume

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('medium')


class CS_Dataset(Dataset):
    def __init__(self,
                 dataset: str,
                 split: str,
                 input: str,
                 target: str,
                 dataset_path: str,
                 transforms=None,
                 preload: bool = True,
                 resample: list[float] | None = None,
                 crop2content: bool = False,
                 padd2same_size: bool = False):
        """Constructor for CS_Dataset class

        Args:
            dataset (str): Dataset name. Only 'bvisa' supported for now. .
            split (str): 'train', 'validation' or 'test' based on the splits.py.

            input (str): Defines value for the 'image' key. One of the following
                'sulci_skeletons' - load  both l/r sulci skeletons (as in BrainVisa paper)
                'left_skeleton' - load left hemisphere sulci skeleton
                'right_skeleton' - load right hemisphere sulci skeleton
                'skull_stripped' - skull-stripped MP-RAGE images.

            target (str, optional): Which image/object to load as target. One of the following:
                'left_sulci' - load left hemisphere sulci labels
                'right_sulci' - load right hemisphere sulci labels
                'all_sulci_bin' - load left and right hemisphere sulci labels and binarize them
                'central_sulcus' - load central sulcus binary labels

            dataset_path (str | None, optional): Path to the dataset folder with the
                directories of subjects in BrainVisa format. Defaults to None.

            transforms (list[callable] | None, optional): List of transforms to apply to the images.
            preload (bool, optional): Preload the images into RAM memory. Defaults to True.
            resample (list[x, y, z] | None, optional): Resample the images to a given resolution.
            crop2content (bool, optional): Crop the images to the content of the image.
            padd2same_size (bool, optional): Pad the images to the same size depending on image type.
        """

        # save dataset hyperparameters
        self.target = target
        self.input = input
        self.dataset = dataset
        self.split = split
        self.dataset_path = Path(dataset_path)
        self.resample = list(resample) if resample is not None else None
        self.preload = preload
        self.transforms = transforms
        self.crop2content = crop2content
        self.padd2same_size = padd2same_size

        # load corresponding image and target paths
        self.img_paths = []
        self.target_paths = []
        self.caseids = []

        if self.dataset == 'bvisa':
            self._load_bvisa()
        else:
            raise ValueError(f'Dataset: {dataset} not Implemented')

    def _load_bvisa(self):
        sulci_path = os.environ.get('SULCI_PATH_PREFIX')
        skeleton_path = os.environ.get('SKELETON_PATH_PREFIX')

        for subj in self.dataset_path.iterdir():
            if subj.is_dir() and subj.name in bvisa_splits[self.split]:

                # get image input paths
                image_paths = []
                if self.input == 'sulci_skeletons' or self.input == 'left_skeleton':
                    lsulci = subj/f'{skeleton_path}/Lskeleton_{subj.name}.nii.gz'
                    image_paths.append(lsulci)

                if self.input == 'sulci_skeletons' or self.input == 'right_skeleton':
                    rsulci = subj/f'{skeleton_path}/Rskeleton_{subj.name}.nii.gz'
                    image_paths.append(rsulci)

                if self.input == 'skull_stripped':
                    image_paths.append(subj/f't1mri/t1/{subj.name}.nii.gz')
                self.caseids.append(image_paths[0].parent.parent.parent.name)
                self.img_paths.append(image_paths)

                # get image target paths
                target_paths = []
                if self.target == 'left_sulci' or self.target == 'all_sulci_bin' \
                        or self.target == 'central_sulcus':
                    lsulci = subj/f'{sulci_path}/LSulci_{subj.name}_base2018_manual.nii.gz'
                    target_paths.append(lsulci)

                if self.target == 'right_sulci' or self.target == 'all_sulci_bin' \
                        or self.target == 'central_sulcus':
                    rsulci = subj/f'{sulci_path}/RSulci_{subj.name}_base2018_manual.nii.gz'
                    target_paths.append(rsulci)

                self.target_paths.append(target_paths)

        # if need to preload store sitk images instead of paths
        if self.preload:
            for i in range(len(self.img_paths)):
                img, trg = self._load_image_target(i)
                self.img_paths[i] = img
                self.target_paths[i] = trg

    def _load_image_target(self, idx):
        # TODO: Fix preloading of the data lag
        """Load image and target from paths"""
        # load target image
        if self.input == 'left_skeleton' or self.input == 'right_skeleton':
            image = sitk.ReadImage(str(self.img_paths[idx][0]))
            image = sitk.Cast(image > 11, sitk.sitkInt16)

        elif self.input == 'sulci_skeletons':
            limage = sitk.ReadImage(str(self.img_paths[idx][0]))
            limage = sitk.Cast(limage > 11, sitk.sitkInt16)

            rimage = sitk.ReadImage(str(self.img_paths[idx][1]))
            rimage = sitk.Cast(rimage > 11, sitk.sitkInt16)

            image = sitk.Cast(((limage + rimage) > 0), sitk.sitkFloat32)

        elif self.input == 'skull_stripped':
            image = sitk.ReadImage(str(self.img_paths[idx][0]))
        else:
            raise ValueError(f'Input: {self.input} not implemented')

        # load target image
        if self.target == 'left_sulci' or self.target == 'right_sulci':
            target = sitk.ReadImage(str(self.target_paths[idx][0]))

        elif self.target == 'all_sulci_bin':
            ltarget = sitk.ReadImage(str(self.target_paths[idx][0])) > 0
            rtarget = sitk.ReadImage(str(self.target_paths[idx][1])) > 0
            target = sitk.Cast((ltarget + rtarget) > 0, sitk.sitkInt16)

        elif self.target == 'central_sulcus':
            ltarget = sitk.ReadImage(str(self.target_paths[idx][0])) == 48
            rtarget = sitk.ReadImage(str(self.target_paths[idx][1])) == 70
            target = sitk.Cast((ltarget + rtarget) > 0, sitk.sitkInt16)
        return image, target

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.preload:
            image, target = self.img_paths[idx], self.target_paths[idx]
        else:
            image, target = self._load_image_target(idx)

        # pre-process images
        image, target = self._preprocess(image, target)

        # post-process images
        image, target = self._postprocess(image, target)


        # apply transforms
        if self.transforms == 'rotate':
            angles = np.zeros(3)  # in radians
            angles[np.random.randint(0, 3)] = np.random.normal(0, np.pi/16)
            rotate = Rotate(angles, keep_size=True,
                            padding_mode='zeros',
                            mode='nearest')
            image = rotate(image).type(torch.float32)  # channels first
            target = rotate(torch.unsqueeze(target, 0))[0].type(torch.int64)

        sample = {'image': image, 'target': target}

        # get caseid
        sample['caseid'] = self.caseids[idx]

        return sample

    def _preprocess(self, image, target):
        """Converts from sitk.Image to torch.Tensor and
           ensures that targets have proper labels and
           images are normalized.

        Args:
            image (sitk.Image): Input image.
            target (sitk.Image): Target image.

        Returns:
            tuple(Tensor, Tensor): Pre-processed image and target.
        """
        # resample if needed
        if self.resample is not None:
            image_interpolator = sitk.sitkLinear if self.input == 'skull_stripped' \
                else sitk.sitkNearestNeighbor
            image = resample_volume(image, self.resample, image_interpolator)
            target = resample_volume(target, self.resample,
                                     interpolator=sitk.sitkNearestNeighbor)
        # convert to numpy
        image = sitk.GetArrayFromImage(image)
        target = sitk.GetArrayFromImage(target)

        # remap labels for the target if more than 1
        if self.target == 'left_sulci':
            new_target = np.zeros_like(target)
            for new_label, orig_label in enumerate(bvisa_left_sulci_labels):
                new_target[target == orig_label] = new_label + 1
            target = new_target
        elif self.target == 'right_sulci':
            new_target = np.zeros_like(target)
            for new_label, orig_label in enumerate(bvisa_right_sulci_labels):
                new_target[target == orig_label] = new_label + 1
            target = new_target

        # crop to content
        if self.crop2content:
            image, min_coords, max_coords = crop_image_to_content(image)
            target, _, __ = crop_image_to_content(target, min_coords, max_coords)

        # min-max normalization of the image
        image = (image - image.min()) / (image.max() - image.min())

        return torch.Tensor(image), torch.tensor(target, dtype=torch.long)

    def _postprocess(self, image: torch.Tensor, target: torch.Tensor):
        # padd if needed
        # TODO: FIX THE ERROR WITH THE ACTIVE PADDING
        if self.padd2same_size:
            size_key = 'original' if self.resample is None else str(self.resample)
            pad_dims = bvisa_padding_dims[self.input][size_key]
            padd = SpatialPad(pad_dims, mode='constant', value=0)
            image = padd(torch.unsqueeze(image, dim=0))[0]
            target = padd(torch.unsqueeze(target, dim=0))[0]

        # add channel dimension to the image
        image = torch.unsqueeze(image, 0)
        return image, target


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
