import cv2 as cv
import numpy as np

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

def feature_matching(left_image, right_image):
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
    
    return src_pts, dst_pts
