import numpy as np
import multiprocessing as mp

import trimesh
import SimpleITK as sitk


def process_sulc_pair(x):
    """Calculates the minimum distance between two sulci
        registered with ICP to each other and the sulcus pair name

    Args:
        x (tuple[list]): sulcus pair, where each sulcus is a list of
            [hemi, subj, type, sitk_segm] as the output of get_subjects()

    Returns:
        tuple: (min_dist, sulcus_pair_name)
    """
    sulc, sulc2 = x
    # register the sulci min(a->b or b->a)

    # need to flip if different hemispheres
    if sulc[0] != sulc2[0]:
        sulc2[3] = sitk.Flip(sulc2[3], [False, False, True])

    # register the sulci with ICP
    s1_points = np.stack(np.where(sitk.GetArrayFromImage(sulc[3]))).T
    s2_points = np.stack(np.where(sitk.GetArrayFromImage(sulc2[3]))).T

    # finds the transformation matrix sending a to b
    _, __, cost_s1_to_s2 = trimesh.registration.icp(a=s1_points, b=s2_points, max_iterations=1000,
                                                    scale=False)
    _, __, cost_s2_to_s1 = trimesh.registration.icp(a=s2_points, b=s1_points, max_iterations=1000,
                                                    scale=False)

    return min(cost_s1_to_s2, cost_s2_to_s1), f'{sulc[0]}_{sulc[1]}-{sulc2[0]}_{sulc2[1]}'


def sulcus_dist(sulc: list, sulclist: list[list]):
    # create a vector of pairwise distances between a given sulcus and all sulci
    dists = []
    reg_keys = []

    sulc_pairs_list = [(sulc, sulc2) for sulc2 in sulclist]

    with mp.Pool(mp.cpu_count()) as p:
        r = list(p.imap(process_sulc_pair, sulc_pairs_list))

    for d, k in r:
        dists.append(d)
        reg_keys.append(k)

    return dists, reg_keys
