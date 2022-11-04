import os
import cv2 as cv
from undistort_image import undistort_image
from get_correspondence_points import calibration, feature_matching
from get_F_matrix import get_F_matrix
from get_E_matrix import get_E_matrix
from decomp_E_matrix import decomp_E_matrix
from estimate_Rrect import estimate_Rrect
from rectify_image import rectify_image
from get_disparity_map import get_disparity_map

left_image_folder_path = 'KITTI-Dataset/2011_09_26_drive_0048/unsync_unrect/image_02/data'
right_image_folder_path = 'KITTI-Dataset/2011_09_26_drive_0048/unsync_unrect/image_03/data'

distorted_list = sorted([file for file in os.listdir(left_image_folder_path)])

left_intrinsic_parameters = {'fx': 9.597910e+02, 'cx': 6.960217e+02, 'fy': 9.569251e+02, 'cy': 2.241806e+02}
left_distortion_parameters = {'k1': -3.691481e-01, 'k2': 1.968681e-01, 'p1': 1.353473e-03, 'p2': 5.677587e-04, 'k3': -6.770705e-02}

right_intrinsic_parameters = {'fx': 9.037596e+02, 'cx': 6.957519e+02, 'fy': 9.019653e+02, 'cy': 2.242509e+02}
right_distortion_parameters = {'k1': -3.639558e-01, 'k2': 1.788651e-01, 'p1': 6.029694e-04, 'p2': -3.922424e-04, 'k3': -5.382460e-02}

if not os.path.exists('left_results'):
    os.makedirs('left_results')
if not os.path.exists('right_results'):
    os.makedirs('right_results')
if not os.path.exists('left_rectified_results'):
    os.makedirs('left_rectified_results')
if not os.path.exists('right_rectified_results'):
    os.makedirs('right_rectified_results')
    
for file_name in distorted_list:
    left_distorted_image = cv.imread(os.path.join(left_image_folder_path, file_name))
    right_distorted_image = cv.imread(os.path.join(right_image_folder_path, file_name))

    undistort_image(file_name, left_distorted_image, left_intrinsic_parameters, left_distortion_parameters, 'left')

    undistort_image(file_name, right_distorted_image, right_intrinsic_parameters, right_distortion_parameters, 'right')

    left_calibration_image = cv.imread('KITTI-Dataset/2011_09_26_drive_0048/calibration/image_02/data/0000000000.png')
    undistort_image('0000000000.png', left_calibration_image, left_intrinsic_parameters, left_distortion_parameters, 'calibration_left')
    intrinsic_parameters_L, distortion_parameters_L, rotation_vectors_L, translation_vectors_L = calibration(left_calibration_image)
    
    right_calibration_image = cv.imread('KITTI-Dataset/2011_09_26_drive_0048/calibration/image_03/data/0000000000.png')
    undistort_image('0000000000.png', right_calibration_image, right_intrinsic_parameters, right_distortion_parameters, 'calibration_right')
    intrinsic_parameters_R, distortion_parameters_R, rotation_vectors_R, translation_vectors_R = calibration(right_calibration_image)

    left_image = cv.imread(os.path.join('left_results', file_name))
    right_image = cv.imread(os.path.join('right_results', file_name))

    left_calib = cv.imread('calibration_left_0000000000.png', 0)
    right_calib = cv.imread('calibration_right_0000000000.png', 0)
    src_pts, dst_pts = feature_matching(left_calib, right_calib)

    F, mask = get_F_matrix(src_pts, dst_pts)

    E = get_E_matrix(intrinsic_parameters_R, intrinsic_parameters_L, F)

    R1, R2, T1, T2 = decomp_E_matrix(E)

    height = left_image.shape[0]
    width = left_image.shape[1]

    H1, H2 = estimate_Rrect(src_pts, dst_pts, F, height, width)

    left_rectified, right_rectified = rectify_image(left_image, right_image, H1, H2, height, width)

    cv.imwrite('left_rectified_results/rectified_left_{}.png'.format(file_name), left_rectified)
    cv.imwrite('right_rectified_results/rectified_right_{}.png'.format(file_name), right_rectified)

    get_disparity_map(left_rectified, right_rectified)