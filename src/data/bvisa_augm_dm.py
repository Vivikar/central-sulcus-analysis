from copy import deepcopy
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
from src.data.bvisa_dm import CS_Dataset as CS_Dataset_NoAugm
from scipy.ndimage import distance_transform_bf
from src.utils.general import crop_image_to_content, resample_volume, sitk_cropp_padd_img_to_size, gaussian_distance_map
logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('medium')


class CS_Dataset(Dataset):
    def __init__(self,
                 dataset: str,
                 split: str,
                 target: str,
                 dataset_path: str,
                 transforms=None,
                 use_half_brain: bool = True,
                 resample: list[float] | None = None,
                 crop2content: bool = False,
                 padd2same_size: bool = False,
                 regresssion: str | None = None,):
        """Constructor for CS_Dataset class

        Args:
            dataset (str): Dataset name. Only 'bvisa' supported for now. .
            split (str): 'train', 'validation' or 'test' based on the splits.py.
            target (str, optional): Which image/object to load as target. One of the following:
                'left_sulci' - load left hemisphere sulci labels
                'right_sulci' - load right hemisphere sulci labels
                'central_sulcus' - load central sulcus binary labels
                'all_sulci_bin' - load all sulci labels but binarize them
            dataset_path (str | None, optional): Path to the dataset folder with the
                directories of subjects in BrainVisa format. Defaults to None.

            transforms (list[callable] | None, optional): List of transforms to apply to the images.
            use_half_brain (bool): Wether you erase a brain hemisphere if target
                is left or right sulci. Defaults to True.
            resample (list[x, y, z] | None, optional): Resample the images to a given resolution.
            crop2content (bool, optional): Crop the images to the content of the image.
            padd2same_size (bool, optional): Pad the images to the same size depending on image type.
        """

        # save dataset hyperparameters
        self.target = target
        self.dataset = dataset
        self.split = split
        self.dataset_path = Path(dataset_path)
        self.resample = list(resample) if resample is not None else None
        self.transforms = transforms
        self.use_half_brain = use_half_brain
        self.regresssion = regresssion

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

        for subj in self.dataset_path.iterdir():
            if subj.is_dir() and subj.name in bvisa_splits[self.split]:

                # get image input paths
                subj_imgs = [x for x in subj.glob('image_t1*.nii.gz')]
                self.img_paths.append(subj_imgs)
                self.caseids.append(subj_imgs[0].name)

                target_paths = [Path(str(x).replace('image', 'labels')) for x in subj_imgs]
                self.target_paths.append(target_paths)


    def _load_image_target(self, idx):

        # load either left, right or both parts of image-label
        """Load image and target from paths"""
        # load a random image from the subject
        rand_idx = np.random.randint(0, len(self.img_paths[idx]))

        # TODO Think how to properly sepearate the image now into left and right
        # bit for now leaving it like this
        image = sitk.ReadImage(str(self.img_paths[idx][rand_idx]))
        orig_target = sitk.ReadImage(str(self.target_paths[idx][rand_idx]))

        # # TODO: Reorient image to looks like Synthseg
        # image = sitk.DICOMOrient(image, 'RIP')
        # orig_target = sitk.DICOMOrient(orig_target, 'RIP') 

        # padd image TODO: remove hardcoding of padding values
        if self.padd2same_size:
            padd_size = [int(x) for x in self.padd2same_size.split('-')]
            image = sitk_cropp_padd_img_to_size(image, padd_size, 0)
            orig_target = sitk_cropp_padd_img_to_size(orig_target, padd_size, 0)
        # print('image', image.GetSize())
        # remove half of the brain
        if self.target in ['left_sulci', 'right_sulci'] and self.use_half_brain:
            image_stripped = sitk.GetArrayFromImage(image)
            right_brainmask = sitk.GetArrayFromImage(orig_target)
            right_brainmask = (((right_brainmask >= 2000) & (right_brainmask < 3000)) |
                               (right_brainmask >= 4000))
            if self.target == 'left_sulci':
                image_stripped[right_brainmask] = 0
            elif self.target == 'right_sulci':
                image_stripped[~right_brainmask] = 0
            image_stripped = sitk.GetImageFromArray(image_stripped)
            image_stripped.CopyInformation(image)
            image = image_stripped

        # split the mask values
        left_sulci, right_sulci = self.remove_btissue_labels(sitk.GetArrayFromImage(orig_target))

        if self.target == 'left_sulci':
            target = sitk.GetImageFromArray(left_sulci)
            target.CopyInformation(orig_target)
        elif self.target == 'right_sulci':
            target = sitk.GetImageFromArray(right_sulci)
            target.CopyInformation(orig_target)
        elif self.target == 'central_sulcus':
            target = sitk.GetImageFromArray(((left_sulci == 48) | (right_sulci == 70)).astype(np.int16))
            target.CopyInformation(orig_target)
        elif self.target == 'all_sulci_bin':
            target = sitk.GetImageFromArray(((left_sulci > 0) | (right_sulci > 0)).astype(np.int16))
            target.CopyInformation(orig_target)
        else:
            raise ValueError(f'Target: {self.target} not implemented')
        return image, target

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image, target = self._load_image_target(idx)

        # pre-process images
        image, target = self._preprocess(image, target)

        # post-process images
        image, target = self._postprocess(image, target)

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
            image_interpolator = sitk.sitkLinear
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

        # apply distance transform
        if self.regresssion == 'distance-transform':
            target = self.dist_transf(target)

        # crop to content
        if self.crop2content:
            image, min_coords, max_coords = crop_image_to_content(image)
            target, _, __ = crop_image_to_content(target, min_coords, max_coords)

        # min-max normalization of the image
        image = (image - image.min()) / (image.max() - image.min())

        if self.regresssion == 'distance-transform':
            return torch.Tensor(image), torch.tensor(target, dtype=torch.float32)
        else:
            return torch.Tensor(image), torch.tensor(target, dtype=torch.long)

    def _postprocess(self, image: torch.Tensor, target: torch.Tensor):
        # padd if needed
        # TODO: FIX THE ERROR WITH THE ACTIVE PADDING
        # if self.padd2same_size:
        #     size_key = 'original' if self.resample is None else str(self.resample)
        #     # TODO: MAKE BETTER PADDING DIMENSIONS SELECTIONS
        #     pad_dims = (256, 256, 256) # bvisa_padding_dims[self.input][size_key]
        #     padd = SpatialPad(pad_dims, mode='constant', value=0)
        #     image = padd(torch.unsqueeze(image, dim=0))[0]
        #     target = padd(torch.unsqueeze(target, dim=0))[0]

        # add channel dimension to the image
        image = torch.unsqueeze(image, 0)

        return image, target

    @staticmethod
    def dist_transf(target: np.ndarray):
        return gaussian_distance_map(target,
                                     alpha=2,
                                     xi=5)

    @staticmethod
    def remove_btissue_labels(img: np.ndarray):
        # remove background tissue labels
        mask1000 = (img>=1000) & (img<2000)
        img[mask1000] = img[mask1000] - 1000

        mask2000 = (img>=2000) & (img<3000)
        img[mask2000] = img[mask2000] - 2000

        mask3000 = (img>=3000) & (img<4000)
        img[mask3000] = img[mask3000] - 3000

        mask4000 = (img>=4000) & (img<5000)
        img[mask4000] = img[mask4000] - 4000

        left_sulci = deepcopy(img)
        left_sulci[np.isin(img, bvisa_right_sulci_labels)] = 0

        right_sulci = deepcopy(img)
        right_sulci[np.isin(img, bvisa_left_sulci_labels)] = 0
        return left_sulci, right_sulci


class CS_DataModule(LightningDataModule):
    def __init__(self,
                 dataset_cfg: dict,
                 train_batch_size: int = 1,
                 validation_batch_size: int = 1,
                 num_workers: int = 1,
                 double_validation: bool = False,
                 ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.num_workers = num_workers
        self.double_validation = double_validation
        self.train_dataset = CS_Dataset(split='train', **dataset_cfg)

        self.val_dataset = CS_Dataset(split='validation', **dataset_cfg)

        bvisa_orig_path = os.environ.get('BVISA_PATH')
        resample = (2, 2, 2) if '2x' in dataset_cfg['dataset'] else None
        self.orig_val_dataset = CS_Dataset_NoAugm(split='validation',
                                                  dataset=dataset_cfg['dataset'],
                                                  target=dataset_cfg['target'],
                                                  input='skull_stripped',
                                                  dataset_path=bvisa_orig_path,
                                                  preload=False,
                                                  resample=resample,
                                                  crop2content=False,
                                                  padd2same_size=False
                                                  )
        logger.info(f'Len of train examples {len(self.train_dataset)} ' +
                    f'len of validation examples {len(self.val_dataset)}')

        self.double_val_ds = DoubleValidationDataset(self.val_dataset, self.orig_val_dataset, self.double_validation)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers)

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.double_val_ds,
            batch_size=self.validation_batch_size,
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader

    def orig_val_dataloader(self):
        val_loader = DataLoader(
            self.orig_val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader


class DoubleValidationDataset(Dataset):
    def __init__(self, dataset: CS_Dataset,
                 orig_dataset: CS_Dataset_NoAugm,
                 double_validation: bool):
        self.dataset = dataset
        self.orig_dataset = orig_dataset
        self.double_validation = double_validation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.double_validation:
            orig_sample = self.orig_dataset[idx]
        else:
            orig_sample = []
        return sample, orig_sample
