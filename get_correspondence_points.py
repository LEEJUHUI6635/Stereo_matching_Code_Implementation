import cv2 as cv
import numpy as np
import os

# Q. Calibration을 하는 이유 : 2D point -> 3D point?
# 하나의 image에 대해서,

# Calibration image -> intrinsic parameter를 구한다. -> intrinsic parameters, radial distortion coefficients of the lens

# print(calib_image.shape) # [512, 1392, 3]

# input : 2D image 좌표와 3D world 좌표계가 알려진 점들을 포함하는 images / output : 3x3 intrinsic parameters, R|t

# Camera Calibration 순서도
# 1. 체커보드 패턴으로 실세계 좌표 정의 -> 3D 점은 체커보드의 코너, X_w, Y_w는 체커보드 위에 있으며, Z_w는 체커보드에 수직이다.
# 2. 다양한 시점에서 체커보드의 여러 이미지 캡처
# 3. 체커보드의 2D 좌표 찾기 -> known : 세계 좌표에서 체커보드에 있는 점의 3D 위치(Z_w = 0), Need to Know : 체커보드 코너의 2D 픽셀 위치
# -> 체커보드 코너 찾기의 내장 함수
# retval, corners = cv.findChessboardCorners(image, patternSize, flags)
# image -> 체커보드 사진 / patternSize -> 체커보드 행과 열 당 내부 코너 수
# 4. 세계 좌표의 3D 점과 모든 이미지의 2D 위치 -> opencv의 내장 함수

# # 체커보드의 차원 정의
# checkerboard1 = (7, 11) # 행, 열
# checkerboard2 = (7, 5) # 행, 열
# checkerboard3 = (5, 7) # 행, 열

# # criteria -> ?
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # 3D 점 벡터를 저장할 벡터 생성
# objpoints = []

# # 2D 점 벡터를 저장할 벡터 생성
# imgpoints = []

# objp = np.zeros((1, checkerboard1[0] * checkerboard1[1], 3), np.float32)
# objp[0,:,:2] = np.mgrid[0:checkerboard1[0], 0:checkerboard1[1]].T.reshape(-1, 2)
# prev_img_shape = None

# # 주어진 디렉토리에 저장된 개별 이미지의 경로 추출
# calib_image = cv.imread('KITTI-Dataset/2011_09_26_drive_0048/calibration/image_00/data/0000000000.png')
# ret, corners = cv.findChessboardCorners(calib_image, checkerboard1, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

# if ret == True:
#     objpoints.append(objp)
#     # corners2 = cv.cornerSubPix(calib_image, corners, (11, 11), (-1, -1), criteria)
#     imgpoints.append(corners)
#     img = cv.drawChessboardCorners(calib_image, checkerboard1, corners, ret)
# #     cv.imshow('img', img)
# #     cv.waitKey(0)
# # cv.destroyAllWindows()
    
# height = calib_image.shape[0]
# width = calib_image.shape[1]

# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, [height, width], cameraMatrix=None, distCoeffs=None)

# print(mtx)
# print(dist)
# print(rvecs, tvecs)

# [[8.09278854e+02 0.00000000e+00 2.74391570e+02]
#  [0.00000000e+00 8.07115450e+03 6.87307353e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# [[-2.78299554e-01  4.96076197e+01 -3.94745566e-02 -5.11041978e-01
#   -1.25947877e+03]]
# (array([[-1.04962304],
#        [-1.04271711],
#        [-1.36807312]]),) (array([[-11.85562476],
#        [ -3.10352319],
#        [ 78.17923206]]),)

# Feature-based RANSAC method -> correspondence matching / ORB feature matching + RANSAC
# Undistorted image
left_path = './left_results/0000000000.png'
right_path = './right_results/0000000000.png'
left_image = cv.imread(left_path)
right_image = cv.imread(right_path)

# SIFT descriptor 생성
detector = cv.ORB_create()

# 각 영상에 대해 keypoint와 descriptor 추출
kp1, desc1 = detector.detectAndCompute(left_image, None)
kp2, desc2 = detector.detectAndCompute(right_image, None)

# matcher 생성, Hamming 거리
matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = matcher.match(desc1, desc2)

# 매칭 결과를 거리기준 오름차순으로 정렬 ---③
matches = sorted(matches, key=lambda x:x.distance)
# 모든 매칭점 그리기 ---④
res1 = cv.drawMatches(left_image, kp1, right_image, kp2, matches, None, \
                    flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# 매칭점으로 원근 변환 및 영역 표시 ---⑤
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

# RANSAC으로 변환 행렬 근사 계산 ---⑥
mtrx, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
h,w = left_image.shape[:2]
pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
dst = cv.perspectiveTransform(pts,mtrx)
right_image = cv.polylines(right_image,[np.int32(dst)],True,255,3, cv.LINE_AA)

# 정상치 매칭만 그리기 ---⑦
matchesMask = mask.ravel().tolist()
res2 = cv.drawMatches(left_image, kp1, right_image, kp2, matches, None, \
                    matchesMask = matchesMask,
                    flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# 모든 매칭점과 정상치 비율 ---⑧
accuracy=float(mask.sum()) / mask.size
print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))

# 결과 출력                    
# cv.imshow('Matching-All', res1)
# cv.imshow('Matching-Inlier ', res2)
# cv.waitKey()
# cv.destroyAllWindows()