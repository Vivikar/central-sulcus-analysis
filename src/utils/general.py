import numpy as np
import SimpleITK as sitk

FreeSurferColorLUT = 'utils/FreeSurferColorLUT.txt'
def fs_lut(lut_path:str=FreeSurferColorLUT):
    """Retrieves the FreeSurfer LUT
    

    Args:
        lut_path (str, optional): Path to the FreeSuerferColorLUT.txt file. Defaults to FreeSurferColorLUT.

    Returns:
        tuple[dict, dict]: fs_lut_names, fs_lut_colors - dictionaries with the
            freesurfer mask labels ask keys and names or colors as values
    """

    with open(lut_path, 'r') as f:
        r=f.readlines()
        
    fs_lut_names = {}
    fs_lut_colors = {}
    for l in r:
        line_data = [x for x in l.split(' ') if x]
        if len(line_data) == 6 and line_data[0].isdigit():
            fs_lut_names[int(line_data[0])] = line_data[1]
            fs_lut_colors[int(line_data[0])] = [int(x.strip()) for x in line_data[2:]]
    return fs_lut_names, fs_lut_colors

def min_max_norm(img):
    """Performs min-max normalization on the input image
        Returns the normalized image as np.uint8
    """
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)


def resample_volume(volume: sitk.Image,
                    new_spacing: list = [1, 1, 1],
                    interpolator = sitk.sitkLinear):
    """Resamples image to a new voxel spacing.
       Results in an image with different dimensions.

    Args:
        volume (sitk.Image): Image to resample
        interpolator (_type_, optional): Interpolator for the voxel sampling. Defaults to sitk.sitkLinear.
        new_spacing (list, optional): New voxel size. Defaults to [1, 1, 1].

    Returns:
        sitk.Image: Image with new voxel spacing
    """
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())


def crop_image_to_content(image: np.ndarray,
                          min_coords: None | tuple = None,
                          max_coords: None | tuple = None):
    """ Crops image to content (non-zero voxels) 
    Args:
        image (np.ndarray): Image to crop
        min_coords (None|tuple, optional): Minimum coordinates to crop. Defaults to None.
        max_coords (None|tuple, optional): Maximum coordinates to crop. Defaults to None.
            if min_coords and max_coords are None, the function will calculate the min and max
            otherwise, the function will use the provided coordinates to crop
    Returns:
        tuple[np.ndarray, tuple, tuple]: Cropped image, min_coords, max_coords
            (coordinates used to crop the given image)
    """
    if min_coords is None or max_coords is None:
        non_zeros = np.stack(np.nonzero(image)).T
        x_min, y_min, z_min = non_zeros.min(0)
        x_max, y_max, z_max = non_zeros.max(0)
    else:
        x_min, y_min, z_min = min_coords
        x_max, y_max, z_max = max_coords

    return (image[x_min:x_max, y_min:y_max, z_min:z_max],
            (x_min, y_min, z_min),
            (x_max, y_max, z_max))


def sitk_cropp_padd_img_to_size(img: sitk.Image, size, padding_value=0):
    """Pads or crops the image to the given size

    Args:
        img (sitk.Image): Image to crop
        size (tuple, optional): Size of the output image in sitk convention.
            Could be the output of img.GetSize(). Defaults to (256, 256, 124).
        padding_value (int, optional): Value to use for padding. Defaults to 0.

    Returns:
        sitk.Image: Center padded/cropped image to given dimensions
    """
    padding_filter = sitk.ConstantPadImageFilter()
    cropping_filter = sitk.CropImageFilter()
    for dim in range(len(img.GetSize())):
        size_origin = [0, 0, 0]
        size_end = [0, 0, 0]
        padding_size = size[dim] - img.GetSize()[dim]

        # padd image
        if padding_size > 0:
            padding_size = abs(padding_size)
            size_origin[dim] = padding_size//2
            size_end[dim] = padding_size - size_origin[dim]
            # padd image from the left
            padding_filter.SetPadLowerBound(size_origin)
            padding_filter.SetPadUpperBound(size_end)
            padding_filter.SetConstant(padding_value)
            img = padding_filter.Execute(img)

        # crop the image
        elif padding_size < 0:
            padding_size = abs(padding_size)
            size_origin[dim] = padding_size//2
            size_end[dim] = padding_size - size_origin[dim]
            cropping_filter.SetLowerBoundaryCropSize(size_origin)
            cropping_filter.SetUpperBoundaryCropSize(size_end)
            
            img = cropping_filter.Execute(img)
    return img
