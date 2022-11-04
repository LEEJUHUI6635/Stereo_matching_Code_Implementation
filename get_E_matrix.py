import numpy as np

def get_E_matrix(intrinsic_R, intrinsic_L, F):
    E = np.transpose(intrinsic_R) @ F @ intrinsic_L
    
    return E