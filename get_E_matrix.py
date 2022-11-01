import numpy as np

# Essential matrix -> left image의 intrinsic parameter의 transpose x fundamental matrix x right image의 intrinsic parameter
def get_E_matrix(intrinsic_R, intrinsic_L, F):
    E = np.transpose(intrinsic_R) @ F @ intrinsic_L
    return E