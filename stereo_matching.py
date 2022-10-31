import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 주어진 parameter로 왜곡 제거 -> 먼저 한 장의 이미지만으로
left_distorted_image = cv.imread('KITTI-Dataset/2011_09_26_drive_0048/unsync_unrect/image_00/data/0000000000.png')
right_distorted_image = cv.imread('KITTI-Dataset/2011_09_26_drive_0048/unsync_unrect/image_01/data/0000000000.png')

# left intrinsic parameters
left_intrinsic_parameters = {'fx': 9.842439e+02, 'cx': 6.900000e+02, 'fy': 9.808141e+02, 'cy': 2.331966e+02}
# left distortion parameters
left_distortion_parameters = {'k1': -3.728755e-01, 'k2': 2.037299e-01, 'p1': 2.219027e-03, 'p2': 1.383707e-03, 'k3': -7.233722e-02}

# right intrinsic parameters
right_intrinsic_parameters = {'fx': 9.895267e+02, 'cx': 7.020000e+02, 'fy': 9.878386e+02, 'cy': 2.455590e+02}
# right distortion parameters
right_distortion_parameters = {'k1': -3.644661e-01, 'k2': 1.790019e-01, 'p1': 1.148107e-03, 'p2': -6.298563e-04, 'k3': -5.314062e-02}

def undistort_image(distorted_image, intrinsic_parameters, distortion_parameters, lr):
    height = distorted_image.shape[0]
    width = distorted_image.shape[1]
    clean_image = np.zeros_like(distorted_image)

    fx = intrinsic_parameters['fx']
    cx = intrinsic_parameters['cx']
    fy = intrinsic_parameters['fy']
    cy = intrinsic_parameters['cy']
    
    k1 = distortion_parameters['k1']
    k2 = distortion_parameters['k2']
    k3 = distortion_parameters['k3']
    p1 = distortion_parameters['p1']
    p2 = distortion_parameters['p2']
    
    for y in range(height):
        for x in range(width):
            y_nu = (y-cy)/fy
            x_nu = (x-cx)/fx

            ru2 = x_nu*x_nu + y_nu*y_nu
            radial_d = 1 + k1*ru2 + k2*ru2*ru2 + k3*ru2*ru2*ru2
            
            x_nd = radial_d*x_nu + 2*p1*x_nu*y_nu + p2*(ru2 + 2*x_nu*x_nu)
            y_nd = radial_d*y_nu + p1*(ru2 + 2*y_nu*y_nu) + 2*p2*x_nu*y_nu

            x_pd = fx*x_nd + cx
            y_pd = fy*y_nd + cy

            x_pd = int(x_pd)
            y_pd = int(y_pd)
            clean_image[y, x] = distorted_image[y_pd, x_pd]
            
    if lr.lower() == 'left':
        cv.imwrite('./left_results/0000000000.png', clean_image)
    elif lr.lower() == 'right':
        cv.imwrite('./right_results/0000000000.png', clean_image)

# # left image에 대한 distortion 제거
# undistort_image(left_distorted_image, left_intrinsic_parameters, left_distortion_parameters, 'left')
# # right image에 대한 distortion 제거
# undistort_image(right_distorted_image, right_intrinsic_parameters, right_distortion_parameters, 'right')

# Calibration
def calibration(calib_image):
    # 체커보드의 차원 정의
    checkerboard = (7, 11) # 행, 열

    # 반복을 종료할 조건 -> (type, max_iter, epsilon)
    # 0.001의 정확도에 도달하면 반복을 중단
    # 30만큼 반복하고 중단
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3D 점 벡터를 저장할 벡터 생성
    objpoints = []

    # 2D 점 벡터를 저장할 벡터 생성
    imgpoints = []

    # object points -> [77, 3], 3 = x + y + z
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    # world coordinate의 z값 = 0 / 나머지는 grid화
    ret, corners = cv.findChessboardCorners(calib_image, checkerboard, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True: # 검출이 되면,
        objpoints.append(objp) # calibration board 상의 3D points
        # corners2 = cv.cornerSubPix(calib_image, corners, (11, 11), (-1, -1), criteria) # grayscale인 경우에만
        imgpoints.append(corners) # calibration board 상의 pixel points
        img = cv.drawChessboardCorners(calib_image, checkerboard, corners, ret)
    #     cv.imshow('img', img)
    #     cv.waitKey(0)
    # cv.destroyAllWindows()
        
    height = calib_image.shape[0]
    width = calib_image.shape[1]

    ret, intrinsic_parameters, distortion_parameters, rotation_vectors, translation_vectors = cv.calibrateCamera(objpoints, imgpoints, [height, width], cameraMatrix=None, distCoeffs=None)
    return intrinsic_parameters, distortion_parameters, rotation_vectors, translation_vectors

# left image calibration
left_calibration_image = cv.imread('KITTI-Dataset/2011_09_26_drive_0048/calibration/image_00/data/0000000000.png')
intrinsic_parameters_L, distortion_parameters_L, rotation_vectors_L, translation_vectors_L = calibration(left_calibration_image)

# right image calibration
right_calibration_image = cv.imread('KITTI-Dataset/2011_09_26_drive_0048/calibration/image_01/data/0000000000.png')
intrinsic_parameters_R, distortion_parameters_R, rotation_vectors_R, translation_vectors_R = calibration(right_calibration_image)

# ORB Feature matching -> 8개의 matching 쌍을 획득

# undistortion -> left image feature matching
left_image = cv.imread('left_results/0000000000.png')
right_image = cv.imread('right_results/0000000000.png')

detector = cv.ORB_create()
kp1, desc1 = detector.detectAndCompute(left_image, None)
kp2, desc2 = detector.detectAndCompute(right_image, None)

# matcher 생성, Hamming 거리
matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = matcher.match(desc1, desc2)

# 매칭 결과를 거리기준 오름차순으로 정렬
matches = sorted(matches, key=lambda x:x.distance)

# 매칭점으로 원근 변환 및 영역 표시
# src_pts -> left image keypoints / dst_pts -> right image keypoints
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

F, mask = cv.findFundamentalMat(src_pts, dst_pts, cv.FM_LMEDS)

# Essential matrix -> left image의 intrinsic parameter의 transpose x fundamental matrix x right image의 intrinsic parameter
# E = np.transpose(provided_R) @ F @ provided_L

# Essential matrix를 분해하여 상대적인 pose를 구해야 한다.
# R1, R2, t = cv.decomposeEssentialMat(E) # Singular Value Decomposition

# Estimate R_rect
h = 1392
w = 512

_, H1, H2 = cv.stereoRectifyUncalibrated(
    src_pts, dst_pts, F, imgSize=(h, w))

# Rectify image
img1_rectified = cv.warpPerspective(left_image, H1, (h, w))
img2_rectified = cv.warpPerspective(right_image, H2, (h, w))

cv.imwrite("rectified_left.png", img1_rectified)
cv.imwrite("rectified_right.png", img2_rectified)

# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# imgL = cv2.imread('KITTI-Dataset/2011_09_26_drive_0048/sync_rect/image_00/data/0000000000.png',0)
# imgR = cv2.imread('KITTI-Dataset/2011_09_26_drive_0048/sync_rect/image_01/data/0000000000.png',0)
# stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15) # numDisparities -> 16의 배수
# disparity = stereo.compute(imgL,imgR)
# filteredImg = cv.normalize(src=disparity, dst=disparity,
#                             beta = 0, alpha = 255, norm_type=cv.NORM_MINMAX)
# plt.imshow(filteredImg,'gray')
# plt.show()
