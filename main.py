from numpy.linalg import eig
import numpy as np
import cv2
import os
import sys
import pickle
from src.extract_features import *
from src.matching_features import *
from src.homography import *
from src.ransac import *
from src.parsing import *
from src.display_video import *

# OVERVIEW:
#   Feature detection: opencv
#   Matching : sklearn , numpy
#   Create Homography: numpy
#   RANSAC: numpy
    
def homography_two_frames(img1, img2, sift_points, kp_list, option):
    """ Compute the homography for two frames """
    match = matching_features(sift_points, img1, img2, cv2.SIFT_create(5000))
    print("Number of matches: ", len(match), '\n')
    if (option == 1):
        src, dst, H  = create_homography_openCV(match, kp_list, 0, 19)
    if (option == 2):
        print(match)
        src1, dst1 = create_src_dest(match, kp_list, 0, 19)
        print("src: ", len(src1), '\n')
        print("dst: ", dst1, '\n')
        H, inliers = RANSAC(src1, dst1, 50, 0.8)
    test_homography(img1, img2, H)
    
def create_all_homographies(match, kp_list):
    """ Compute the homography for all frames """
    i = 0
    size = len(match)
    matrix_H = [[None for _ in range(size)] for _ in range(size)] 
    while (match[i]):
        j = 0
        while (match[j]):
            print(match[i][j])
            matrix_H[i][j] = create_homography(match[i][j], kp_list, i, j)
            matrix_H[i][j], inliers = RANSAC(src, dst, 50, 0.8)
            j += 1
        i += 1
    return matrix_H

def create_sequential_homographies(matches, sift_points):
    H_sequential=np.empty([11,0])
    
    for i in range(len(matches)):
        match=matches[i]
        kp_src = sift_points[i][:2,:]
        kp_dst = sift_points[i+1][:2,:]

        src_pts = np.float32([ kp_src[:,q] for q in match[0,:]])
        dst_pts = np.float32([ kp_src[:,q] for q in match[0,:]])

        H_parameters, inliers = RANSAC(src_pts, dst_pts, 50, 0.8)

        H=np.transpose(np.array([i+1, i+2, H_parameters]))
        H_sequential = np.hstack([H_sequential, H])
    return H_sequential

if __name__ == "__main__":

    if (check_syntax()):
        sys.exit(1)
    config_data = parse_configuration_file(sys.argv[1]) #Parse the configuration file
    match_img1 , match_map = parse_points(config_data) #Parse the points from the configuration file
    video_path = config_data[0].split(' ')[1].strip() #Get the video path
    H_frame1_to_map =compute_homography(match_img1, match_map)
    print(H_frame1_to_map)
    print("Condition: ", np.linalg.cond(H_frame1_to_map), '\n')
    
    sift_points, kp_list, img1, img2 = extract_features(video_path)
    homography_two_frames(img1, img2, sift_points, kp_list, 2) #option 1 - with openCV; option 2 - with numpy
    
    #match2 = matching_features_SCIKITLEARN(sift_points)
    #print(match2)
    
    #H_sequential = create_sequential_homographies(match2, sift_points)

    #matrix_H = create_all_homographies(match2, kp_list)
    #output_file_path = 'path/file_for_transforms.ext'
    #with open(output_file_path, 'wb') as file:
        #pickle.dump(matrix_H, file)
    
# Condition -> A*A^(-1) - High condition means small changes in the input can result in large changes in the output


#WHAT IS MATCH?
#match = {[[D_matchFrame1WithFrame2_1, .. DmatchFrame1WithFrame2_N], [D_matchFrame1WithFrameN, .. DmatchFrame1WithFrameN]],
#                                                   .....
#        [[D_matchFrameNWithFrame1_1, .. DmatchFrameNWithFrame1_N], [D_matchFrameNWithFrameN_1, .. DmatchFrameNWithFrameN_N]]}
# match[i] = list of all matches of frame i with all other frames
# match[i][j] = list of all matches of frame i with frame j