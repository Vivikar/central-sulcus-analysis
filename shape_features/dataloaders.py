from pathlib import Path

import open3d as o3d
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset
from tqdm import tqdm


class VIA11_Corrected_CS_Loader(Dataset):
    def __init__(self,
                 bv_good: bool = True,
                 corrected: bool = True,
                 all_bv: bool = False,
                 preload: bool = False) -> None:
        """Constructor for VIA11_Corrected_CS_Loader

        Args:
            bv_good (bool, optional): Flag to indicate wether to load BrainVISA segmentations
                that are good (type 1). Defaults to True.
            corrected (bool, optional): Flag to indicate wether to load the manually
                corrected CS segmentations. Overrides any BrainVISA segmentations loaded. Defaults to True.
            all_bv (bool, optional): Flag to indicate wether to load all BrainVISA segmentations no matter their QC status.
                Overrides all other flags. Defaults to False.
            preload: (bool, optional): Flag to indicate wether to preload the data in the 
                _get_item() or to return the paths to images. Defaults to False.
        """

        self.bv_good = bv_good
        self.corrected = corrected
        self.all_bv = all_bv
        self.preload = preload

        # store segmentations in format 
        # {subject_id: sub-viaxxx, type': type_of_segmentation},
        #  'lsegm': path_to_left_segmentation, 'rsegm': path_to_right_segmentation, 
        #  'lmesh': path_to_left_mesh, 'rmesh': path_to_right_mesh}
        #               '}
        self.cs_segmentations = []
        self._load_data()

    def __len__(self):
        return len(self.cs_segmentations)

    def __getitem__(self, idx):
        if self.preload:
            d = self.cs_segmentations[idx]
            return {'subject_id': d['subject_id'],
                    'type': d['type'],
                    'lsegm': sitk.ReadImage(d['lsegm']),
                    'rsegm': sitk.ReadImage(d['rsegm']),
                    'lmesh': o3d.io.read_triangle_mesh(d['lmesh']),
                    'rmesh': o3d.io.read_triangle_mesh(d['rmesh'])}
        else:
            return self.cs_segmentations[idx]

    def _load_data(self):
        corrected_loaded = []
        if self.corrected:
            corr_paths = Path('/mnt/projects/VIA_Vlad/nobackup/BrainVisa/CS_edited').glob('**/LSu*sub-via*.nii.gz')
            for lsegmpath in corr_paths:
                resgmpath = str(lsegmpath).replace('LSulci', 'RSulci')
                lmeshpath = str(lsegmpath).replace('.nii.gz', '.ply')
                rmeshpath = str(lmeshpath).replace('LSulci', 'RSulci')

                subj = lsegmpath.name[7:17]
                subj_data = {'subject_id': subj}
                subj_data['lsegm'] = lsegmpath
                subj_data['rsegm'] = resgmpath
                subj_data['lmesh'] = lmeshpath
                subj_data['rmesh'] = rmeshpath
                subj_data['type'] = 'corrected'
                subj_data['site'] = str(lsegmpath).split('/')[-2]
                self.cs_segmentations.append(subj_data)
                corrected_loaded.append(subj)

        if self.bv_good:
            # load QC results
            qc_results = pd.read_excel(Path('/mnt/projects/VIA_Vlad/nobackup/QA_centralSulcus_lkj.xlsx'))
            qc_results = qc_results.set_index('subjects')
            bv_good_subjs = qc_results[qc_results.vis_QA == 1]

            for subj, metadata in bv_good_subjs.iterrows():

                # skip BrainVISA segmentations if corrected segmentations are loaded
                if subj in corrected_loaded:
                    continue

                lsegmpath = f'/mnt/projects/VIA_Vlad/nobackup/BrainVisa/BrainVisa/{metadata.sites}/{subj}/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_best/segmentation/LSulci_{subj}_default_session_best.nii.gz'
                resgmpath = lsegmpath.replace('LSulci', 'RSulci')
                lmeshpath = lsegmpath.replace('.nii.gz', '.ply')
                rmeshpath = lmeshpath.replace('LSulci', 'RSulci')

                subj_data = {'subject_id': subj}
                subj_data['lsegm'] = lsegmpath
                subj_data['rsegm'] = resgmpath
                subj_data['lmesh'] = lmeshpath
                subj_data['rmesh'] = rmeshpath
                subj_data['type'] = 'bv_good'
                subj_data['site'] = metadata.sites
                self.cs_segmentations.append(subj_data)

        if self.all_bv:
            self.cs_segmentations = []
            # load QC results
            qc_results = pd.read_excel(Path('/mnt/projects/VIA_Vlad/nobackup/QA_centralSulcus_lkj.xlsx'))
            qc_results = qc_results.set_index('subjects')
            bv_good_subjs = qc_results[qc_results.vis_QA != 999]

            for subj, metadata in bv_good_subjs.iterrows():

                lsegmpath = f'/mnt/projects/VIA_Vlad/nobackup/BrainVisa/BrainVisa/{metadata.sites}/{subj}/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_best/segmentation/LSulci_{subj}_default_session_best.nii.gz'
                resgmpath = lsegmpath.replace('LSulci', 'RSulci')
                lmeshpath = lsegmpath.replace('.nii.gz', '.ply')
                rmeshpath = lmeshpath.replace('LSulci', 'RSulci')

                subj_data = {'subject_id': subj}
                subj_data['lsegm'] = lsegmpath
                subj_data['rsegm'] = resgmpath
                subj_data['lmesh'] = lmeshpath
                subj_data['rmesh'] = rmeshpath
                subj_data['type'] = 'bv_good'
                subj_data['site'] = metadata.sites
                self.cs_segmentations.append(subj_data)

    def get_subjects(self):
        """Returns a list of all subjects in the dataset

        Returns:
            list: ['left|right', 'sub-viaxxx', 'corrected|bv', 'sitk_image']
        """
        sulci = [self.__getitem__(i) for i in tqdm(range(len(self.cs_segmentations)))]

        # unravell all the sulci
        sulci_list = []
        for s in sulci:
            lsegm = s['lsegm'] if self.preload else sitk.ReadImage(s['lsegm'])
            rsegm = s['rsegm'] if self.preload else sitk.ReadImage(s['rsegm'])

            sulci_list.append(['left', s['subject_id'], s['type'],
                               lsegm])
            sulci_list.append(['right', s['subject_id'], s['type'],
                               rsegm])
        return sulci_list
