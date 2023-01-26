from pathlib import Path

# TODO: Move to environment file! ####################
BRAIN_VISA_PATH = "/mnt/projects/VIA_Vlad/nobackup/BrainVisa/BrainVisa/"
CS_CORRECTED = '*/sub-via*/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_best/segmentation/*cleaned.nii*'
SEGM_PATH = 't1mri/default_acquisition/default_analysis/folds/3.1/default_session_best/segmentation'

#########################

corrected_paths = Path(BRAIN_VISA_PATH)
corrected_paths = [x for x in corrected_paths.glob(CS_CORRECTED)]

