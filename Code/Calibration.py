"""
    This script generates the matrix camera calibration:
    it takes in input some chessboard images, which are used to correct the lens distortion.
"""
import numpy as np
import cv2
import yaml

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane


for i in range(1,18):
    string1='../Sources/Chessboard/position'
    string2=str(i)
    string3='.png'
    filename_input=string1+string2+string3
    img = cv2.imread(filename_input)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (5,7),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (5,7), corners2,ret)
        # Display the resulting frame
        smallimage = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        cv2.imshow('img', smallimage)
        # Save the resulting frame
        string1='../Output/Chessboard/Corners/position'
        string2=str(i)
        string3='.png'
        image_name=string1+string2+string3
        cv2.imwrite(image_name,img)

        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibration example on the first, second and third chessboard image
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
for index in range(1,18):
    string1='../Sources/Chessboard/position'
    string2=str(index)
    string3='.png'
    filename_input=string1+string2+string3
    img = cv2.imread(filename_input)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    string1='../Output/Chessboard/Results/calibrresult_position'
    string2=str(index)
    string3='.png'
    filename_output=string1+string2+string3
    cv2.imwrite(filename_output,dst)

# Save camera matrix on a file
data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}

with open("calibration.yaml", "w") as f:
    yaml.dump(data, f)