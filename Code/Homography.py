"""
    This script computes the homography matrix between the video and the image of a basketball fiels.
"""

import cv2
import copy
import numpy as np
import yaml

# Open the video
cap = cv2.VideoCapture('../Output/Video/undistorted.primo_tempo.asf')
# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read the image
img = cv2.imread('../Sources/Map/basket_field.jpg')

# Initialize the vectors for point acquisition
# Homography requires at least 4 points

# Lists used for by the function draw_circle
x_sequence=list()
y_sequence=list()

# Lists used for the points in the video
x_sequence_video=list()
y_sequence_video=list()

# List used for the points in the image
x_sequence_image=list()
y_sequence_image=list()

# mouse callback function
# you can draw a circle with a left click and the point coordinates will be saved in two lists:
#       1) x_sequence for x coordinates
#       2) y_sequence for y coordinates
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
         cv2.circle(param,(x,y),4,(255,0,0),-1)
         x_sequence.append(x)
         y_sequence.append(y)

# Point acquisition from the first frame of the video
if cap.isOpened():
    ret, frame = cap.read()
    smallFrame = cv2.resize(frame, (0, 0), fy=0.35, fx=0.35)
    height, width, channels = smallFrame.shape
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', draw_circle, smallFrame)
    while (1):
        cv2.imshow('frame', smallFrame)
        # To end the point acquisition click 'q'
        if cv2.waitKey(20) & 0xFF == ord('q'):
            # Copy of the points in the lists for the video
            x_sequence_video=copy.copy(x_sequence)
            y_sequence_video=copy.copy(y_sequence)
            break

# Clean the actual lists of the function in order to start the acquisition for the image
del x_sequence[:]
del y_sequence[:]
# Point acquisition in the image
while (1):
    cv2.setMouseCallback('image', draw_circle, img)
    cv2.imshow('image', img)
    # To end the point acquisition click 'q'
    if cv2.waitKey(20) & 0xFF == ord('q'):
        # Copy of the points in the lists for the image
        x_sequence_image=copy.copy(x_sequence)
        y_sequence_image=copy.copy(y_sequence)
        break
cv2.destroyAllWindows()

# Create point_source and point_destination
# We must have the same number of points for video and image to compute homography
if len(x_sequence_video)!=len(x_sequence_image):
    print("Error homography: number of source points and destination points must agree")
# We must check to have at least 4 points
elif len(x_sequence_video)>=4:
    print("Homography computation")

    # Create point vectors for video and image
    points_source=np.column_stack([x_sequence_video,y_sequence_video])
    print("x sequence match:")
    print(x_sequence_video)
    print("y sequence match:")
    print(y_sequence_video)
    print("points source:")
    print(points_source)
    points_destination=np.column_stack([x_sequence_image,y_sequence_image])
    print("x sequence basket:")
    print(x_sequence_image)
    print("y sequence basket:")
    print(y_sequence_image)
    print("points destination:")
    print(points_destination)

    # Creation of the homography matrix
    h, status = cv2.findHomography(points_source, points_destination)
    print("homography matrix:")
    print(h)

    # Save the homography matrix on a file
    data = {'homography': np.asarray(h).tolist()}
    with open("homography_11points.yaml", "w") as f:
        yaml.dump(data, f)
