import numpy as np
import multiprocessing as mp

import trimesh
import SimpleITK as sitk

from scipy.ndimage import map_coordinates

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


def interpolate_segmentation(imageA:np.ndarray,
                             imageB_shape:tuple,
                             pointsAregistered2B:np.ndarray,
                             trnsh_mat:np.ndarray,
                             spline_order: int=1,
                             threshold: float=0.3,
                             offset: int = 2):
    """Interpolates the segmentation of the image A to the image B space

    Args:
        imageA (np.ndarray): Image containing the segmentation in the space A
        imageB_shape (tuple): Shape of the output image B
        pointsAregistered2B (np.ndarray): Coordinates of the segmentation
            points from the image A registered to image B space
        trnsh_mat (np.ndarray): Transformation matrix used to register
            the segmentation from the image A to the image B space.
            [[Rx, Ry, Rz, Tx], [Rx, Ry, Rz, Ty], [Rx, Ry, Rz, Tz], [0, 0, 0, 1]]
        spline_order (int, optional): Order of the spline used for
            interpolation. If 0 - nearest neighbor, if 1-linear interpolation.
            Defaults to 1.
        threshold (float, optional): Threshold used to binarize
            the interpolated segmentation. Defaults to 0.3.
        offset (int, optional): Offset in the bounds of the ROI.
            Since for faster execution interpolation is done only inside the ROI
            of the imageB that contains the segmentation. Defaults to 2.

    Returns:
        np.ndarray: imageB containing the interpolated segmentation
            defined by the pointsAregistered2B
        
    """
    
    # find the boundaries of the registered segmentation ROI
    # (in the image B space) as only the values inside this ROI
    # +- offset will be interpolated for faster computation
    # as no values are expected to be found far away from the ROI
    xmin = int(np.min(pointsAregistered2B[:, 0])) - offset
    xmax = int(np.max(pointsAregistered2B[:, 0])) + offset
    ymin = int(np.min(pointsAregistered2B[:, 1])) - offset
    ymax = int(np.max(pointsAregistered2B[:, 1])) + offset
    zmin = int(np.min(pointsAregistered2B[:, 2])) - offset
    zmax = int(np.max(pointsAregistered2B[:, 2])) + offset

    # placeholder for the resulting image
    A_registered_to_B = np.zeros(imageB_shape)

    # get an array with ROI point coordinates for which to interpolate
    A_registered_to_B[xmin:xmax, ymin:ymax, zmin:zmax] = -1
    points2interp = np.stack(np.where(A_registered_to_B == -1)).T

    # transform the points of the image A_registered_to_B
    # back to the image A space
    points2interp = np.dot(np.column_stack((points2interp,
                                        np.ones(points2interp.shape[0]))),
                        np.linalg.inv(trnsh_mat).T)[:, :3]
    
    # perform the interpolation in the image A space
    interp_vals = map_coordinates(imageA, points2interp.T,
                                  order=spline_order, mode='constant',
                                  cval=0, output=np.float32)
    
    # threshold the resulting values to get a binary image
    interp_vals = (interp_vals>threshold).astype(np.int16)
    interp_vals = interp_vals.reshape((xmax-xmin, ymax-ymin, zmax-zmin))
    # fill the ROI with the values from the registered image A
    A_registered_to_B[xmin:xmax, ymin:ymax, zmin:zmax] = interp_vals
    
    return A_registered_to_B