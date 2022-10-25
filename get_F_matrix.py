import cv2 as cv
import numpy as np
import get_correspondence_points as corr

# Epiploar Geometry -> 8-point algorithms

# 앞서 구한 feature matching 중 8 쌍의 feature를 가져온다.

pts1 = []
pts2 = []

for i, (m, n) in enumerate(corr.matches):
    if m.distance < 0.8*n.distance:
        pts2.append(corr.kp2[m.trainIdx].pt)
        pts1.append(corr.kp1[m.queryIdx].pt)
        
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
print(F)
