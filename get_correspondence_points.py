import cv2 as cv
import numpy as np

def calibration(calib_image):
    checkerboard = (7, 11) 
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = []
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    ret, corners = cv.findChessboardCorners(calib_image, checkerboard, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        img = cv.drawChessboardCorners(calib_image, checkerboard, corners, ret)
    height = calib_image.shape[0]
    width = calib_image.shape[1]
    ret, intrinsic_parameters, distortion_parameters, rotation_vectors, translation_vectors = cv.calibrateCamera(objpoints, imgpoints, [height, width], cameraMatrix=None, distCoeffs=None)
    
    return intrinsic_parameters, distortion_parameters, rotation_vectors, translation_vectors

def feature_matching(left_image, right_image):
    detector = cv.ORB_create()
    kp1, desc1 = detector.detectAndCompute(left_image, None)
    kp2, desc2 = detector.detectAndCompute(right_image, None)
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x:x.distance)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    return src_pts, dst_pts