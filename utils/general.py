import numpy as np

def min_max_norm(img):
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

def get_corss3d_SE():
    se_cross3d = np.zeros((3, 3, 3))
    se_cross3d[0, 1, 1] = 1
    se_cross3d[1, :, 1] = 1
    se_cross3d[1, 1, :] = 1

    se_cross3d[2, 1, 1] = 1
    return se_cross3d