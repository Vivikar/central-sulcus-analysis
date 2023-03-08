import os
import numpy as np
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator
from pathlib import Path


# Path to save the generated images and to the labels which to use
target_path = Path('/mrhome/vladyslavz/git/central-sulcus-analysis/data/synthseg/nobackup')
training_label_maps = Path('/mrhome/vladyslavz/git/SynthSeg/data/training_label_maps')

# script parameters
n_examples = 100  # number of examples to generate in this script
result_dir = './outputs_tutorial_5'  # folder where examples will be saved


# path training label maps
path_label_map = '/mrhome/vladyslavz/git/SynthSeg/data/training_label_maps'
generation_labels = '/mrhome/vladyslavz/git/SynthSeg/data/labels_classes_priors/generation_labels.npy'
output_labels = '/mrhome/vladyslavz/git/SynthSeg/data/labels_classes_priors/synthseg_segmentation_labels.npy'
n_neutral_labels = 18
output_shape = None  # INSTEAD OF RANDOM CROPPING USER ORIGINAL IMAGES
# output_shape = 192
# target_res = 256


# ---------- GMM sampling parameters ----------

# Here we use Gaussian priors to control the means and standard deviations of the GMM.
prior_distributions = 'normal'

# Here we still regroup labels into classes of similar tissue types:
# Example: (continuing the example of tutorial 1)  generation_labels = [0, 24, 507, 2, 3, 4, 17, 25, 41, 42, 43, 53, 57]
#                                                 generation_classes = [0,  1,   2, 3, 4, 5,  4,  6,  3,  4,  5,  4,  6]
# Note that structures with right/left labels are now associated with the same class.
generation_classes = '/mrhome/vladyslavz/git/SynthSeg/data/labels_classes_priors/generation_classes_contrast_specific.npy'

# We specify here the hyperparameters governing the prior distribution of the GMM.
# As these prior distributions are Gaussian, they are each controlled by a mean and a standard deviation.
# Therefore, the numpy array pointed by prior_means is of size (2, K), where K is the total number of classes specified
# in generation_classes. The first row of prior_means correspond to the means of the Gaussian priors, and the second row
# correspond to standard deviations.
#
# Example: (continuing the previous one) prior_means = np.array([[0, 30, 80, 110, 95, 40, 70]
#                                                                [0, 10, 50,  15, 10, 15, 30]])
# This means that intensities of label 3 and 17, which are both in class 4, will be drawn from the Gaussian
# distribution, whose mean will be sampled from the Gaussian distribution with index 4 in prior_means N(95, 10).
# Here is the complete table of correspondence for this example:
# mean of Gaussian for label   0 drawn from N(0,0)=0
# mean of Gaussian for label  24 drawn from N(30,10)
# mean of Gaussian for label 507 drawn from N(80,50)
# mean of Gaussian for labels 2 and 41 drawn from N(110,15)
# mean of Gaussian for labels 3, 17, 42, 53 drawn from N(95,10)
# mean of Gaussian for labels 4 and 43 drawn from N(40,15)
# mean of Gaussian for labels 25 and 57 drawn from N(70,30)
# These hyperparameters were estimated with the function SynthSR/estimate_priors.py/build_intensity_stats()
prior_means = '/mrhome/vladyslavz/git/SynthSeg/data/labels_classes_priors/prior_means_t1.npy'
# same as for prior_means, but for the standard deviations of the GMM.
prior_stds = '/mrhome/vladyslavz/git/SynthSeg/data/labels_classes_priors/prior_stds_t1.npy'

# ---------- Resolution parameters ----------

# here we aim to synthesise data at a specific resolution, thus we do not randomise it anymore !
randomise_res = False

# blurring/downsampling parameters
# We specify here the slice spacing/thickness that we want the synthetic scans to mimic. The axes refer to the *RAS*
# axes, as all the provided data (label maps and images) will be automatically aligned to those axes during training.
# RAS refers to Right-left/Anterior-posterior/Superior-inferior axes, i.e. sagittal/coronal/axial directions.
data_res = np.array([1., 1., 1.])  # slice spacing i.e. resolution to mimic
thickness = np.array([1., 1., 1.])  # slice thickness

# ---------- Spatial augmentation ----------

# We now introduce some parameters concerning the spatial deformation. They enable to set the range of the uniform
# distribution from which the corresponding parameters are selected.
# We note that because the label maps will be resampled with nearest neighbour interpolation, they can look less smooth
# than the original segmentations.

flipping = False  # enable right/left flipping # IMPORTANT AS WE WANT TO COMPARE CORTEX SHAPES FROM THE SAME AND DIFFERENT SUBJECTS
scaling_bounds = 0.11  # the scaling coefficients will be sampled from U(1-scaling_bounds; 1+scaling_bounds)
rotation_bounds = 10  # the rotation angles will be sampled from U(-rotation_bounds; rotation_bounds)
shearing_bounds = 0.01  # the shearing coefficients will be sampled from U(-shearing_bounds; shearing_bounds)
translation_bounds = False  # no translation is performed, as this is already modelled by the random cropping
nonlin_std = 3.  # this controls the maximum elastic deformation (higher = more deformation)
bias_field_std = 0.5  # his controls the maximum bias field corruption (higher = more bias)



# ------------------------------------------------------ Generate ------------------------------------------------------

for lm in training_label_maps.glob('*.nii.gz'):
    seg_name = lm.name[:-7]  # training_seg_01 and lm is full path
    print('Started processing: ', seg_name, ' label map...')

    # instantiate BrainGenerator object per each label map
    brain_generator = BrainGenerator(labels_dir=path_label_map,
                                     generation_labels=generation_labels,
                                     output_labels=output_labels,
                                     n_neutral_labels=n_neutral_labels,
                                     output_shape=output_shape,
                                     #  target_res=target_res,
                                     prior_distributions=prior_distributions,
                                     generation_classes=generation_classes,
                                     prior_means=prior_means,
                                     prior_stds=prior_stds,
                                     randomise_res=randomise_res,
                                     data_res=data_res,
                                     thickness=thickness)
    # make a directory for each label map
    if not (target_path/seg_name).exists():
        (target_path/seg_name).mkdir(exist_ok=True, parents=True)

    for n in range(n_examples):
        print(target_path/seg_name/f'image_t1_{n}.nii.gz')
        # generate new image and corresponding labels
        im, lab = brain_generator.generate_brain()

        # save output image and label map
        utils.save_volume(im, brain_generator.aff, brain_generator.header,
                          str(target_path/seg_name/f'image_t1_{n}.nii.gz'))
        utils.save_volume(lab, brain_generator.aff, brain_generator.header,
                          str(target_path/seg_name/f'labels_t1_{n}.nii.gz'))
