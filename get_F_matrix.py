# Eight-point Algorithm

import cv2
import numpy as np
import matplotlib.pyplot as plt

# queryImage = left image
img1 = cv2.imread('./left_results/0000000000.png',0) 
img1 = cv2.resize(img1,(400,400))
# trainImage = right image
img2 = cv2.imread('./right_results/0000000000.png',0)
img2 = cv2.resize(img2,(400,400))





# ratio test
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)





# sift = cv2.SIFT_create()

# # SIFT로 keypoints와 descriptors 찾기
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)

# # FLANN 파라미터
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE,
#                    tress = 5)
# search_params = dict(checks=50)

# flann = cv2.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des1,des2,k=2)

# good = []
# pts1 = []
# pts2 = []

# # ratio test
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.8*n.distance:
#         good.append(m)
#         pts2.append(kp2[m.trainIdx].pt)
#         pts1.append(kp1[m.queryIdx].pt)

# pts1 = np.int32(pts1)
# pts2 = np.int32(pts2)
# F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

# print(F)


# # inlier 포인트만 선택한다.
# pts1 = pts1[mask.ravel()==1]
# pts2 = pts2[mask.ravel()==1]

# def drawlines(img1,img2,lines,pts1,pts2):
#     "img1 : img2의 점에 대한 epiline을 그리는 이미지"
#     "lines : epilines에 대응하는 것"
#     r, c = img1.shape
#     img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#     img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
#     for r, pt1, pt2 in zip(lines,pts1,pts2): 
#         color = tuple(np.random.randint(0,255,3).tolist())
#         x0,y0 = map(int,[0,-r[2]/r[1]])
#         x1,y1 = map(int,[c,-(r[2]+r[0]*c)/r[1]])
#         img1 = cv2.line(img1,(x0,y0),(x1,y1),color,1)
#         img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
#         img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
#     return img1,img2

# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2),2,F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# # 왼쪽 이미지의 포인트에 해달하는 epilines 찾기
# # 오른쪽 이미지에 선 그리기
# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2),1,F)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

# cv2.imshow('left image with line', img5)
# cv2.imshow('left image points', img6)
# cv2.imshow('right image with line', img3)
# cv2.imshow('right image points', img4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
