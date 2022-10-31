import os
import cv2 as cv
from undistort_image import undistort_image
from get_correspondence_points import calibration, feature_matching

left_results = './left_results'
right_results = './right_results'

# if not os.path.exists(left_results):
#     os.makedirs(left_results)
# if not os.path.exists(right_results):
#     os.makedirs(right_results)

# left_folder = 'KITTI-Dataset/2011_09_26_drive_0048/unsync_unrect/image_00/data'
# right_folder = 'KITTI-Dataset/2011_09_26_drive_0048/unsync_unrect/image_01/data'

# left_distorted = sorted([file for file in os.listdir(left_folder)])
# right_distorted = sorted([file for file in os.listdir(right_folder)])

# left_distorted_images = [cv.imread(os.path.join(left_folder, file)) for file in left_distorted]
# right_distorted_images = [cv.imread(os.path.join(right_folder, file)) for file in right_distorted]

# left_intrinsic_parameters = {'fx': 9.842439e+02, 'cx': 6.900000e+02, 'fy': 9.808141e+02, 'cy': 2.331966e+02}
# left_distortion_parameters = {'k1': -3.728755e-01, 'k2': 2.037299e-01, 'k3': 2.219027e-03, 'p1': 1.383707e-03, 'p2': -7.233722e-02}

# right_intrinsic_parameters = {'fx': 9.895267e+02, 'cx': 7.020000e+02, 'fy': 9.878386e+02, 'cy': 2.455590e+02}
# right_distortion_parameters = {'k1': -3.644661e-01, 'k2': 1.790019e-01, 'k3': 1.148107e-03, 'p1': -6.298563e-04, 'p2': -5.314062e-02}

# # Left image
# undistort_image(left_distorted_images, left_intrinsic_parameters, left_distortion_parameters, 'left')

# # Right image
# undistort_image(right_distorted_images, right_intrinsic_parameters, right_distortion_parameters, 'right')

# Calibration
left_calib_image = cv.imread('KITTI-Dataset/2011_09_26_drive_0048/calibration/image_00/data/0000000000.png')
right_calib_image = cv.imread('KITTI-Dataset/2011_09_26_drive_0048/calibration/image_01/data/0000000000.png')

L_intrinsic_parameters, L_distortion_parameters, L_rotation_vectors, L_translation_vectors = calibration(left_calib_image)
R_intrinsic_parameters, R_distortion_parameters, R_rotation_vectors, R_translation_vectors = calibration(right_calib_image)

# print(L_intrinsic_parameters)
# print(L_distortion_parameters)

# [[8.09278854e+02 0.00000000e+00 2.74391570e+02]
#  [0.00000000e+00 8.07115450e+03 6.87307353e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

# [[-2.78299553e-01  4.96076197e+01 -3.94745566e-02 -5.11041978e-01
#   -1.25947877e+03]]

# print(R_intrinsic_parameters)
# print(R_distortion_parameters)

# [[1.2458570e+03 0.0000000e+00 2.4697444e+02]
#  [0.0000000e+00 2.4570771e+03 6.1469469e+02]
#  [0.0000000e+00 0.0000000e+00 1.0000000e+00]]
# [[-1.09675113e+00  4.03194500e+01  3.61955871e-01 -6.16714172e-02
#   -2.00713748e+02]]

# Correspondence matching
# Feature-based RANSAC method
left_path = sorted([file for file in os.listdir(left_results)])

for path in left_path:
    left_path = os.path.join(left_results, path)
    right_path = os.path.join(right_results, path)
    
    feature_matching(left_path, right_path)
    
    # Epipolar Geometry
    
    # get_F_matrix
    
    # get_E_matrix
    
    # decomp_E_matrix
    
    # rectify_image
    
    # get_disparity_map