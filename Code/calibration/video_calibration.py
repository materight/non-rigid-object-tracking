"""
    This script generates the calibrated version of the video using the camera matrix
    and save it in a file.
"""
import numpy as np
import cv2
import yaml

# Load the camera matrix from file
with open('configs/calibration.yaml') as f:
    loadeddict = yaml.load(f)

mtxloaded = loadeddict.get('camera_matrix')
distloaded = loadeddict.get('dist_coeff')
mtx=np.asarray(mtxloaded)
dist=np.asarray(distloaded)

# Read the video
cap = cv2.VideoCapture('../Sources/primo_tempo.asf')

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Set output video
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('../Output/undistorted.primo_tempo.asf', fourcc, 20.0, (3840, 2160))

# Undistort the video frame by frame
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # Save undistorted video
        out.write(dst)
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
