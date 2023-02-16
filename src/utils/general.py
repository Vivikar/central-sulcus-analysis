import numpy as np


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

