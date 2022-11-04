import cv2 as cv
import numpy as np

def decomp_E_matrix(E):
    u, s, v_t = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    T1 = u @ W @ s @ np.transpose(u)
    T2 = u @ np.transpose(W) @ s @ np.transpose(u)
    R1 = u @ np.linalg.inv(W) @ v_t
    R2 = u @ W @ v_t
    
    return T1, T2, R1, R2