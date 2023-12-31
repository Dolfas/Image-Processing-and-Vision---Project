import numpy as np
from numpy.linalg import eig
import cv2
from src.ransac import *
from src.matching_features import *
from sklearn.neighbors import NearestNeighbors


def matching_direct_homography(features_frame_i,features_frame_j):
    Threshold=0.75
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(features_frame_j) 
    distances, indices = nbrs.kneighbors(features_frame_i)     
    features_matches=np.empty([4,0])

    for i in range(len(distances)): 
        if distances[i,0]< Threshold*distances[i,1]:
            #match is good for first neighbour found
            features_matches= np.hstack((  features_matches   , np.array([[int(i)],[int(indices[i,0])], [distances[i,0]],[distances[i,1]]])  ))


    features_matches = features_matches[:, features_matches[1, :].argsort()] # this sorts the check_for_duplicates matrix in accordance to the values of it's second line
    features_matches_deletedColumns= features_matches.copy()

    for i in reversed (range (1, features_matches.shape[1])): #loop that starts in the last feature - because it deletes elements with their indexes from list check_for_duplicates_deletedColumns
        # this has to be done starting from the end to not change the index of columns

        # duplicates are adjacent because of sort
        if features_matches[1,i-1] == features_matches[1,i]:
            # if the value of the indice i and i-1 are equal, then there is one feature matched to 2 features of the new frame - we need to delete one of the matches
            if features_matches[2,i-1] <= features_matches[2,i]: #check distance of i and i-1. And remove the one with the most distance
                features_matches_deletedColumns= np.delete(features_matches_deletedColumns, i-1, 1) #remove duplicate feature matching (deletes one column - np dimension 1)
            else:
                features_matches_deletedColumns= np.delete(features_matches_deletedColumns, i, 1) 
    
    matched_inThis_frame = features_matches_deletedColumns[:, features_matches_deletedColumns[0, :].argsort()] #to be in order in acoordance to index of frame s

    return matched_inThis_frame[0:2,:]

def recalculate_1_homography_if_intersection(H,height, width, sift_points, index_frame_dst, index_frame_src):
    #Coordinates of the corners
    transformed_corners = calculate_transformed_corners(H, width, height) # corners coordinates in destination frame (frame 8)
    # Check if all points are on one side of the centerline horizontally or vertically
    center_x, center_y = width // 2, height // 2
    all_above = all(c[1] < center_y for c in transformed_corners)
    all_below = all(c[1] > center_y for c in transformed_corners)
    all_right = all(c[0] > center_x for c in transformed_corners)
    all_left = all(c[0] < center_x for c in transformed_corners)
    
    # Recompute homography with filtered features
    if not (all_above or all_below or all_right or all_left):
        #Features of frames
        features_frame_i = np.transpose(sift_points[int(index_frame_src)][2:,:])  # Features from frame 1 (origin)
        features_frame_j = np.transpose(sift_points[int(index_frame_dst)][2:,:]) # Features from frame 8 (destination)   
        
        matches= matching_direct_homography(features_frame_i,features_frame_j)

        kp_dst = sift_points[index_frame_dst][:2,:] #keyupoints for frame n (destination)
        kp_src= sift_points[index_frame_src][:2,:] #keypoints for frame n+1 (origin)

        src_pts = [] #[[x1,y1, d1, ..., dn], [x1,y1, d1, ..., dn], ...]
        dst_pts = []
        for k in matches[0,:]:
            src_pts.append(  (float(kp_src[0,int(k)])   , float(kp_src[1,int(k)]))   )
        for k in matches[1,:]:
            dst_pts.append(  (float(kp_dst[0,int(k)])   , float(kp_dst[1,int(k)]))   )

        # Now src_pts_list_of_lists and dst_pts_list_of_lists contain lists of lists
        output_H, _ = RANSAC(src_pts, dst_pts, 72, 0.8)
    else:
        output_H= H
    return output_H

def calculate_transformed_corners(H, width, height):
    # Corners of the image in homogenous coordinates
    corners = np.array([
        [0, 0, 1], # Top left
        [width - 1, 0, 1], # Top right
        [width - 1, height - 1, 1], # Bottom right
        [0, height - 1, 1] # Bottom left
    ]).T
    
    # Apply the homography to the corners
    transformed_corners = H @ corners #this 
    transformed_corners /= transformed_corners[2] # Normalize the points
    #print("coners", transformed_corners[:2].T)
    return transformed_corners[:2].T

def find_original_coordinates(H_inv, transformed_corners):    
    # Apply the inverse homography to the transformed corners
    original_corners = H_inv @ np.vstack([transformed_corners.T, np.ones((1, transformed_corners.shape[0]))])
    original_corners /= original_corners[2] # Normalize the points
    return original_corners[:2].T

def is_point_inside_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def filter_features_outside_corners(keypoints, corners):
    # keypoints is a numpy array where each row is a keypoint and the first two columns are x and y coordinates.
    inside_keypoints = []
    for kp in keypoints:
        x, y = kp[0], kp[1]  # Extract the (x, y) coordinates from the keypoint
        if is_point_inside_polygon((x, y), corners):
            inside_keypoints.append((x, y))

    return inside_keypoints
