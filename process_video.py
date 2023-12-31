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
from src.view_homography import *
from src.outputs import *
from src.optimized_corners import *

# OVERVIEW:
#   Feature detection: opencv
#   Matching : sklearn , numpy
#   Create Homography: numpy
#   RANSAC: numpy

def all_homographies(H_sequential):
    """ This function computes the homography from any frame i to any frame j (Hij), with j>i
    This means that we are doing homography in the direction of a bigger frame, which is contrary to the direction of the H_sequential homographies
    To solve this, the H_sequential matrixes must be inverted before being used"""

    H_output=np.empty([11,0])
    
    for i in range(1,H_sequential.shape[1]+1): #H_sequential[0] is H_21
        for j in range(i+1, H_sequential.shape[1] +2):
            #the homographie between frame j-1 and j is:

            H_jminus1_j = np.linalg.inv(H_sequential[2:,j-2].reshape((3,3))) #H_sequential is H from 2 to 1, from 3 to 2, from 4 to 3... and we want the inverse direction
            
            if i+1 == j: #simple homographie
                T_to_map=  H_jminus1_j
            else: #compound of sequential homographies
                T_to_map= np.matmul( H_jminus1_j , H_output[2:,-1].reshape(3,3) ) # example: frame4 = H34*H23*H12*frame1

            H_i = np.vstack((np.array([[j],[i]] ), T_to_map.reshape(9,1) ))
            H_output = np.hstack([H_output, H_i])
        
    return H_output

def homography_to_map(H_sequential, H_frame1_to_map):
    """ This function computes the homography from any frame to the map.
        For frame n, H_output[2:,i-1] should be the homography from frame n-1 to the map. 
        H_sequential[2:,i-1] should be the homography from frame n to n-1
        So T_to_Map should be the homography from frame n to map"""
    H_output=np.empty([11,0])
    H_i = np.vstack((np.array([[0], [1]]) , H_frame1_to_map.reshape(9,1) )) #first part of the array is 0 and 1 - which means homography from frame 1 to map (frame 0)
    H_output = np.hstack([H_output, H_i])
    
    for i in range(1, H_sequential.shape[1]):
        T_to_map= np.matmul(H_output[2:,i-1].reshape(3,3), H_sequential[2:,i-1].reshape(3,3)) 
        H_i = np.vstack((np.array([[0],[H_sequential[1,i-1]]] ), T_to_map.reshape(9,1) ))
        H_output = np.hstack([H_output, H_i])
        
    return H_output

def create_sequential_homographies(matches, sift_points):
    """ This function creates the homographies from frame n+1 to frame n.
        We want the homography from frame n+1 to frame n because we will 
        want the homography from any frame back to the frame 1 and then to the map."""
    H_sequential=np.empty([11,0])
    for i in range(len(matches)):
        kp_dst = sift_points[i][:2,:] #keyupoints for frame n (destination)
        kp_src= sift_points[i+1][:2,:] #keypoints for frame n+1 (origin)
        src_pts = [] #[[x1,y1, d1, ..., dn], [x1,y1, d1, ..., dn], ...]
        dst_pts = []
        for k in matches[i][0,:]:
            src_pts.append(  (float(kp_src[0,int(k)])   , float(kp_src[1,int(k)]))   )
        for k in matches[i][1,:]:
            dst_pts.append(  (float(kp_dst[0,int(k)])   , float(kp_dst[1,int(k)]))   )

        H_parameters, inliers = RANSAC(src_pts, dst_pts, 72, 0.8) #71.36 iterations to 0.99 suceess-> for 50% inliers
        indexes_frames = np.array([[i+1], [i+2]])
        H = np.vstack((indexes_frames, H_parameters.reshape(9,1) ))
        H_sequential = np.hstack([H_sequential, H])
        
    return H_sequential

def final_parsing_process_video(config_data):
    video_path = config_data[0].split(' ')[1].strip() #Get the video path
    type_homography = config_data[4].split(' ')[3] #Get the type of homography
    file_name_keypoints = "outputs/" + config_data[3].split(' ')[1] #Get the name of the keypoints file
    file_name_tranformations = "outputs/" + config_data[5].split(' ')[1] #Get the name of the transformations file
    return video_path, type_homography, file_name_keypoints, file_name_tranformations
    
if __name__ == "__main__":
    if (check_syntax()):
        sys.exit(1)
    config_data = parse_configuration_file(sys.argv[1]) #Parse the configuration file
    match_img1 , match_map = parse_points(config_data) #Parse the points from the configuration file
    video_path, type_homography, file_name_keypoints, file_name_tranformations = final_parsing_process_video(config_data)
    
    sift_points, nr_points = extract_features(video_path)
    
    create_output_keypoints(sift_points, file_name_keypoints, nr_points)
    
