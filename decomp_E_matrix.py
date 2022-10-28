from get_E_matrix import E
import cv2 as cv

R1, R2, t = cv.decomposeEssentialMat(E) # Singular Value Decomposition

print(R1, R2, t)