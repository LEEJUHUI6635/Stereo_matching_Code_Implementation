import cv2 as cv

def decomp_E_matrix(E):
    R1, R2, t = cv.decomposeEssentialMat(E) # Singular Value Decomposition
    return R1, R2, t