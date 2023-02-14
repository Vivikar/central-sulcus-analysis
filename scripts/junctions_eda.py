import cv2
import numpy as np
from skimage import morphology, measure
from utils.general import min_max_norm
from utils.mesh import get_corss3d_SE
from tqdm import tqdm
import SimpleITK as sitk
import pandas as pd
from data.cs_image_loader import CS_Images

# Junction templates in 2D used to find junctions of the CS
junc1 = [[0, -1,  0],
         [1,  1,  1],
         [-1,  1,  0]]

junc2 = [[0, -1,  0],
		 [1,  1,  1],
		 [0,  1, -1]]

junc3 = [[1, -1, -1],
         [-1, 1, 1],
         [-1, 1, -1]]

JUNCTION_FBANK = [np.rot90(junc1, k) for k in range(4)]
JUNCTION_FBANK += [np.rot90(junc2, k) for k in range(4)]
JUNCTION_FBANK += [np.rot90(junc3, k) for k in range(4)]

def find_junctions_slice(imgslice:np.ndarray, junctions_fbank=JUNCTION_FBANK):
    """Finds junctions in a 2D image and returns a binary mask with their locations.

    Args:
        imgslice (np.ndarray): np.uint8 image slice with a single channel.
        junctions_fbank (list[np.ndarray], optional): Filter bank with junction template 
            with structuring elements used to look for junctions. Defaults to JUNCTION_FBANK.

    Returns:
        np.ndarray: Boolean mask with junctions locations.
    """
    res = np.zeros_like(imgslice)
    
    # perform thinning of the image
    imgslice_thinned = cv2.ximgproc.thinning(imgslice)
    
    for f in junctions_fbank:
        res += cv2.morphologyEx(imgslice_thinned, cv2.MORPH_HITMISS, f)
    return res > 0

def junction_image(image):
    junction_img = np.zeros_like(image)
    for sl in range(image.shape[0]):
        if np.any(image[sl, :, :]>0):
            junction_img[sl, :, :] = find_junctions_slice(min_max_norm(image[sl, :, :]))
        else:
            junction_img[sl, :, :] = 0
        
    cross3d = get_corss3d_SE()

    # dilate to connect junctions 
    junction_img_dilated = morphology.binary_dilation(junction_img, cross3d)
    junction_img_dilated_labeled = measure.label(junction_img_dilated, connectivity=2)

    junctions_merged = (junction_img>0).astype(np.uint8)*junction_img_dilated_labeled

    return junctions_merged

def junction_analysis():
    dataset = CS_Images(segmentation='brainvisa',
                        mesh=True,
                        preload=False)
    res = []
    figm = dataset[0]['img']
    avg_img = np.zeros_like(sitk.GetArrayFromImage(figm)).astype(np.float64)
    avg_junc = np.zeros_like(sitk.GetArrayFromImage(figm)).astype(np.float64)
    avg_sulci = np.zeros_like(sitk.GetArrayFromImage(figm)).astype(np.float64)
    
    for i in tqdm(range(len(dataset))):
        s0 = dataset[i]
        
        img = sitk.GetArrayFromImage(s0['img'])
        if img.shape[0] != avg_img.shape[0]:
            img = img.T
        avg_img += img/img.max()
        
        bvisa = sitk.GetArrayFromImage(s0['bvisa'])
        if bvisa.shape[0] != avg_sulci.shape[0]:
            bvisa = bvisa.T
        avg_sulci += bvisa/bvisa.max()
        
        lCS_junction_img = junction_image((bvisa==1).astype(np.uint8))
        rCS_junction_img = junction_image((bvisa==2).astype(np.uint8))
        avg_junc += lCS_junction_img + rCS_junction_img
        
        lstats = dict(zip(*np.unique(lCS_junction_img, return_counts=True)))
        lstats = {k:v for k,v in lstats.items() if k!=0 and v>1}
        
        rstats = dict(zip(*np.unique(rCS_junction_img, return_counts=True)))
        rstats = {k:v for k,v in rstats.items() if k!=0 and v>1}

        caseid = s0['caseid']
        centre = s0['centre']
        
        case_results = {'caseid': caseid,
                        'centre': centre,
                        'lCS_junctions': lstats,
                        'rCS_junctions': rstats}
        
        res.append(case_results)
    avg_img = avg_img/len(dataset)
    avg_junc = avg_junc/len(dataset)
    avg_sulci = avg_sulci/len(dataset)
    res_df = pd.DataFrame(res)
    res_df.to_csv('results/junction_analysis.csv', index=False)
    res_df.to_pickle('results/junction_analysis.pkl')
    avg_img = sitk.GetImageFromArray(avg_img)
    avg_img.CopyInformation(figm)
    sitk.WriteImage(avg_img, 'results/avg_img.nii.gz')
    avg_junc = sitk.GetImageFromArray(avg_junc)
    avg_junc.CopyInformation(figm)
    sitk.WriteImage(avg_junc, 'results/avg_junc.nii.gz')
    avg_sulci = sitk.GetImageFromArray(avg_sulci)
    avg_sulci.CopyInformation(figm)
    sitk.WriteImage(avg_sulci, 'results/avg_sulci.nii.gz')
    
    
if __name__ == '__main__':
    junction_analysis()
    