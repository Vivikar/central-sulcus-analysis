"""Contains functions and utilities for mesh processing and analysis
"""
import numpy as np

def get_corss3d_SE():
    """Returns the structuring element for 3D cross
        as np.array of shape (3, 3, 3)"""
    se_cross3d = np.zeros((3, 3, 3))
    se_cross3d[0, 1, 1] = 1
    se_cross3d[1, :, 1] = 1
    se_cross3d[1, 1, :] = 1

    se_cross3d[2, 1, 1] = 1
    return se_cross3d


def dihedral_angle(n1, n2):
    """Dihedral angle ϕ of an edge to be the angle between the
        normals of the two adjacent faces, such that the angle is 
        0 iff the two faces are coplanar but not coinciding and is
        approaching π for sharper angles. It can never become π,
        as this would mean that the two faces are coinciding,
        which is not allowed in a non degenerate mesh. It can also
        never be greater than π, as we always consider the
        smallest angle between the two faces
        
    Args:
        n1 (np.ndarray): Normal of the first face
        n2 (np.ndarray): Normal of the second face
    where n1 and n2 are normals of the two adjacent faces
    
    Returns:
        float: Dihedral angle between the two faces in degrees
            We assume that the normals of the triangles are
            either all pointing outwards or inwards wrt. the mesh.
            This way, ϕ is 0, iff the two triangles are coplanar
            but not coinciding, π/2, if they form a right angle
            regardless of the orientation (concave or convex) and π,
            if they are coinciding, which cannot happen in a mesh
            that is not degenerate. This means that ϕ ∈ [0, π)
    """
    sintheta = np.linalg.norm(np.cross(n1, n2))/(np.linalg.norm(n1)*np.linalg.norm(n2))
    costheta = np.dot(n1, n2)/(np.linalg.norm(n1)*np.linalg.norm(n2))
    
    if costheta>0:
        theta = np.arcsin(sintheta)
    elif costheta<0:
        theta = np.pi - np.arcsin(sintheta)
    else:
        theta = np.pi/2
    return (theta/np.pi)*180