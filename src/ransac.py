import numpy as np
from numpy.linalg import eig
from src.homography import *
from numpy.linalg import svd
      
def RANSAC(src,dst,iter,threshold):
      best_homography = None
      inliers = 0
      best_inliers_arr = np.array([])
      for t in range(iter):
            win = (len(src) if len(src) < len(dst) else len(dst))
            if (win < 4):
                  continue
            sample_indices = np.random.choice(int(win), size=4, replace=False) #purpose of replace=False is to not choose the same index twice
            src_homography = [src[j] for j in sample_indices] #get the points from the random indices
            dst_homography = [dst[j] for j in sample_indices]
            H = compute_homography(src_homography,dst_homography)
            inl = 0
            inliers_arr = []
            for p, q in zip(src, dst): #zip -> iterate through two lists at the same time
                  x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
                  transformed_point = np.dot(H, np.array([x1, y1, 1])) # Transform the point using the estimated homography
                  if transformed_point[2] > -10^(-9) and transformed_point[2] < 10^(-9):
                        transformed_point[2] += 10^(-10)
                  transformed_point /= transformed_point[2] # Normalize the transformed point
                
                  # Calculate the Euclidean distance between the transformed point and the actual point
                  distance = np.linalg.norm(np.array([x2, y2, 1]) - transformed_point)
                
                  if distance < threshold:
                      inliers_arr.append([x1, y1, x2, y2])
                      inl += 1
            inliers_arr = np.array(inliers_arr)
            if inl > inliers:
                  best_inliers_arr = inliers_arr
                  inliers = inl
      best_homography = compute_homography(best_inliers_arr[:, :2], best_inliers_arr[:, 2:4])
      return best_homography, inliers
