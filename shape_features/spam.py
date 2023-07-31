import SimpleITK as sitk
import numpy as np
import trimesh
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.manifold import Isomap

class SPAM:
    def __init__(self, sulci_list: list[list],
                 sulci_distance_matrix: np.ndarray,
                 average_sulci: list | None = None,
                 ) -> None:
        """Constructor for SPAM

        Args:
            sulci_list (list[list]): List of sulci, where each sulcus is a list of
                [hemi, subj, type, sitk_segm] as the output of get_subjects().
            sulci_distance_matrix (np.ndarray): Distance matrix of the sulci in sulci_list.
                Dimensions: (len(sulci_list), len(sulci_list)).
            average_sulci (list[list], optional): Sulcus list that will be used as registration template.
                Usually the sulcus with the smallest average distance to over sulci is chosen.
                If None, calculates the average sulcus from the sulci_list. Defaults to None."""

        self.sulci_list = sulci_list
        
        self.sulci_distance_matrix = sulci_distance_matrix

        if average_sulci is None:
            idx = np.argmin(sulci_distance_matrix.mean(axis=1))
            print(f'Average sulcus automatically selected is {sulci_list[idx][0]}_{sulci_list[idx][1]}.')
            self.set_average_sulcus(sulci_list[idx])
        else:
            self.set_average_sulcus(average_sulci)

    def set_average_sulcus(self, sulcus: list):
        """Changes the sulci from the sulci_list
            to be registered to the given average sulcus
        """
        self.average_sulcus = sulcus

        # register the sulci to the average sulcus
        print('Registering sulci to average sulcus')
        registered_sulci = []
        for sulc2 in tqdm(self.sulci_list):

            # need to flip if different hemispheres
            if sulcus[0] != sulc2[0]:
                sulc2[3] = sitk.Flip(sulc2[3], [False, False, True])

            # register the sulci with ICP
            s1_points = np.stack(np.where(sitk.GetArrayFromImage(sulcus[3]))).T
            s2_points = np.stack(np.where(sitk.GetArrayFromImage(sulc2[3]))).T

            # finds the transformation matrix sending a to b
            _, transformed, __ = trimesh.registration.icp(a=s2_points, b=s1_points, max_iterations=1000,
                                                          scale=False)

            transformed_img = np.zeros_like(sitk.GetArrayFromImage(sulc2[3]))
            transformed_img[tuple(np.round(transformed).astype(int).T)] = 1
            transformed_img = sitk.GetImageFromArray(transformed_img.astype(np.float32))

            res_sulci = sulc2.copy()
            res_sulci[3] = transformed_img
            registered_sulci.append(res_sulci)

        self.sulci_list = registered_sulci

    def retrieve_isomap_spams(self,
                             sample_shapes_n: int = 10,
                             isomap_components: int = 10,
                             n_neighbors: int = 10,
                             l: float = 20,
                             top_prcn = 0.1,):
        # fitting ISOMAP with the sulci distance matrix (all to all)
        self.iso = Isomap(n_components=isomap_components, n_jobs=-1, n_neighbors=n_neighbors, radius=None)
        self.sdm_trasformed = self.iso.fit_transform(self.sulci_distance_matrix)

        isomap_feat_values = []
        all_spam_sulci = []
        # go for each ISOMAP feature and sample uniformly from it
        for i in tqdm(range(self.sdm_trasformed.shape[1])):
            spam_sulci = []
            isomap_feat = self.sdm_trasformed[:, i]

            # sample uniformly from the feature space
            sample_idx = np.linspace(min(isomap_feat),
                                     max(isomap_feat),
                                     sample_shapes_n)

            isomap_feat_values.append(sample_idx)
            # calculate the distance to all sulci for given sampled point from the feature space
            for v in sample_idx:
                dist = (isomap_feat - v)**2
                spam_sulci.append(self.get_spam(dist,
                                                l=l,
                                                top_prcn=top_prcn)
                                  )
            all_spam_sulci.append(spam_sulci)
        return all_spam_sulci, isomap_feat_values

    def get_spam(self, sulci_distances: np.ndarray,
                 l: float = 20,
                 top_prcn = 0.1,):
        """Get the Statistical Probability Anatomical Map sulcus

        Args:
            sulci_distances (np.ndarray): Distances of the ISMAP feature from each sulci in the sulci_list to
                a given value along that ISOMAP feature dimension.
            l (float, optional): Exponential weighting  parameter.
                The smaller it is - the smaller is the contribution of the far-away sulci
                to the SPAM model. Defaults to 100.
            sigma (float, optional): Gaussian smoothing parameter.
                Bigger sigma - bigger smoothing used for SPAM estimation. Defaults to 1.

        Returns:
            np.ndarray: SPAM sulcus
        """

        # perform exponential weighting and normalize the weights-distances
        # for top_prcn of the closet sulci
        dists_topprcn_idxs = np.argsort(sulci_distances)[:int(len(sulci_distances)*top_prcn)+1][1:]
        dists_topprcn = np.sort(sulci_distances)[:int(len(sulci_distances)*top_prcn)+1][1:]
        dists_topprcn = np.exp(-dists_topprcn/l)
        dists_topprcn = dists_topprcn / np.sum(dists_topprcn) # normalizing to sum to 1

        # placeholder for the SPAM sulcus      
        spam_sulcus = sitk.Image(*sitk.GetArrayFromImage(self.sulci_list[0][3]).shape[::-1],
                                 sitk.sitkFloat32)
        spam_sulcus.CopyInformation(self.sulci_list[0][3])

        for idx, dist_weight in enumerate(dists_topprcn):
            spam_sulcus += self.sulci_list[dists_topprcn_idxs[idx]][3] * dist_weight

        spam_sulcus = sitk.GetArrayFromImage(spam_sulcus)

        return spam_sulcus
    def __str__(self) -> str:
        return f'SPAM with {len(self.sulci_list)} sulci and average sulcus {self.average_sulcus[0]}_{self.average_sulcus[1]}'
