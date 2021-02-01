"""
    This script use the homography matrix to create a correspondence between a point in the video and one in the image.
    Then it gives you the position in meters, considering as (0,0) the point at the top left
"""

import cv2
import numpy as np
import yaml
import copy

# Variable for the point coordinates
ix, iy = -1,-1
# font for printing
font = cv2.FONT_HERSHEY_SIMPLEX

# Open a window
cv2.namedWindow('frame')

# Open the video
cap = cv2.VideoCapture('../Output/Video/undistorted.primo_tempo.asf')
# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read the image
img = cv2.imread('../Sources/Map/basket_field.jpg')

# Create copy of image for the different homography
img_4points=copy.copy(img)
img_11points=copy.copy(img)
img_19points=copy.copy(img)

# Read homography matrix
# 1) Homography 4 points
with open('homography_4points.yaml') as f:
    loadeddict = yaml.load(f)

hloaded = loadeddict.get('homography')
h_4points=np.asarray(hloaded)

# 2) Homography 11 points
with open('homography_11points.yaml') as f:
    loadeddict = yaml.load(f)

hloaded = loadeddict.get('homography')
h_11points=np.asarray(hloaded)

# 3) Homography 19 points
with open('homography_19points.yaml') as f:
    loadeddict = yaml.load(f)

hloaded = loadeddict.get('homography')
h_19points=np.asarray(hloaded)

# Open a file for MSE evaluation
MSE_file=open("../Output/Accuracy/MSE_accuracy.txt","w")

# Draw horizontal lines on the video
if cap.isOpened():
    ret, frame = cap.read()
    # Reduce dimension of frame in order to show it on the screen
    smallFrame = cv2.resize(frame, (0, 0), fy=0.35, fx=0.35)
    # Draw an horizontal line to evaluate the accuracy of our homography matrix
    cv2.line(smallFrame,(180,500),(1200,500),(255,0,0),4)
    cv2.line(smallFrame, (335, 400), (1065, 400), (0, 255, 0), 4)
    cv2.line(smallFrame, (255, 450), (1130, 450), (0, 0, 255), 4)
    cv2.imshow('frame', smallFrame)
    cv2.waitKey(0)

# Case 1: blue line
x_line=np.arange(180,1201)
y_line=np.full(1021,500,int)
# Initialization of the lists for the points
x_line_homography_4points=list()
y_line_homography_4points=list()
x_line_homography_11points=list()
y_line_homography_11points=list()
x_line_homography_19points=list()
y_line_homography_19points=list()

# Compute the homography of the line
# Subcase 1: 4 points homography
for i in range(0,1020):
    vector = np.dot(h_4points, np.transpose([x_line[i], y_line[i], 1]))
    x_img = vector[0]
    y_img = vector[1]
    w = vector[2]
    x_new = int(x_img / w)
    y_new = int(y_img / w)
    x_line_homography_4points.append(x_new)
    y_line_homography_4points.append(y_new)
    cv2.circle(img_4points, (x_new, y_new), 4, (255, 0, 0), -1)
    cv2.imshow('image_4points', img_4points)
    # Press 'q' to close the image and then choose another point on the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Subcase 2: 11 points homography
for i in range(0,1020):
    vector = np.dot(h_11points, np.transpose([x_line[i], y_line[i], 1]))
    x_img = vector[0]
    y_img = vector[1]
    w = vector[2]
    x_new = int(x_img / w)
    y_new = int(y_img / w)
    x_line_homography_11points.append(x_new)
    y_line_homography_11points.append(y_new)
    cv2.circle(img_11points, (x_new, y_new), 4, (255, 0, 0), -1)
    cv2.imshow('image_11points', img_11points)
    # Press 'q' to close the image and then choose another point on the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Subcase 3: 19 points homography
for i in range(0,1020):
    vector = np.dot(h_19points, np.transpose([x_line[i], y_line[i], 1]))
    x_img = vector[0]
    y_img = vector[1]
    w = vector[2]
    x_new = int(x_img / w)
    y_new = int(y_img / w)
    x_line_homography_19points.append(x_new)
    y_line_homography_19points.append(y_new)
    cv2.circle(img_19points, (x_new, y_new), 4, (255, 0, 0), -1)
    cv2.imshow('image_19points', img_19points)
    # Press 'q' to close the image and then choose another point on the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Computation of the mean square error:
#   4 points  --> 11 points (MSE_4_11_blue)
#   11 points --> 19 points (MSE_11_19_blue)
#   19 points --> 4 points  (MSE_19_4_blue)

MSE_4_11_x=0
MSE_11_19_x=0
MSE_19_4_x=0
MSE_4_11_y=0
MSE_11_19_y=0
MSE_19_4_y=0

for i in range(0,1020):
    MSE_4_11_x=MSE_4_11_x+(x_line_homography_4points[i]-x_line_homography_11points[i])**2
    MSE_4_11_y = MSE_4_11_y + (y_line_homography_4points[i] - y_line_homography_11points[i])**2
    MSE_11_19_x=MSE_11_19_x+(x_line_homography_11points[i]-x_line_homography_19points[i])**2
    MSE_11_19_y = MSE_4_11_y + (y_line_homography_11points[i] - y_line_homography_19points[i])**2
    MSE_19_4_x=MSE_19_4_x+(x_line_homography_19points[i]-x_line_homography_4points[i])**2
    MSE_19_4_y = MSE_19_4_y + (y_line_homography_19points[i] - y_line_homography_4points[i])**2

MSE_4_11_x=MSE_4_11_x/len(x_line_homography_4points)
MSE_4_11_y=MSE_4_11_y/len(y_line_homography_4points)
MSE_11_19_x=MSE_11_19_x/len(x_line_homography_11points)
MSE_11_19_y=MSE_11_19_y/len(y_line_homography_11points)
MSE_19_4_x=MSE_19_4_x/len(x_line_homography_19points)
MSE_19_4_y=MSE_19_4_y/len(y_line_homography_19points)

# Write on a file the results
MSE_file.write("[BLUE] Mean Square Error homography 4 points --> homography 11 points: (")
MSE_file.write(str(round(MSE_4_11_x,2)))
MSE_file.write(",")
MSE_file.write(str(round(MSE_4_11_y,2)))
MSE_file.write(")\n")
MSE_file.write("[BLUE] Mean Square Error homography 11 points --> homography 19 points: (")
MSE_file.write(str(round(MSE_11_19_x,2)))
MSE_file.write(",")
MSE_file.write(str(round(MSE_11_19_y,2)))
MSE_file.write(")\n")
MSE_file.write("[BLUE] Mean Square Error homography 19 points --> homography 4 points: (")
MSE_file.write(str(round(MSE_19_4_x,2)))
MSE_file.write(",")
MSE_file.write(str(round(MSE_19_4_y,2)))
MSE_file.write(")\n")

# Case 2: green line
x_line=np.arange(335,1066)
y_line=np.full(731,400,int)
# Initialization of the lists for the points
x_line_homography_4points=list()
y_line_homography_4points=list()
x_line_homography_11points=list()
y_line_homography_11points=list()
x_line_homography_19points=list()
y_line_homography_19points=list()

# Compute the homography of the line
# Subcase 1: 4 points homography
for i in range(0,730):
    vector = np.dot(h_4points, np.transpose([x_line[i], y_line[i], 1]))
    x_img = vector[0]
    y_img = vector[1]
    w = vector[2]
    x_new = int(x_img / w)
    y_new = int(y_img / w)
    x_line_homography_4points.append(x_new)
    y_line_homography_4points.append(y_new)
    cv2.circle(img_4points, (x_new, y_new), 4, (0, 255, 0), -1)
    cv2.imshow('image_4points', img_4points)
    # Press 'q' to close the image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Subcase 2: 11 points homography
for i in range(0,730):
    vector = np.dot(h_11points, np.transpose([x_line[i], y_line[i], 1]))
    x_img = vector[0]
    y_img = vector[1]
    w = vector[2]
    x_new = int(x_img / w)
    y_new = int(y_img / w)
    x_line_homography_11points.append(x_new)
    y_line_homography_11points.append(y_new)
    cv2.circle(img_11points, (x_new, y_new), 4, (0, 255, 0), -1)
    cv2.imshow('image_11points', img_11points)
    # Press 'q' to close the image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Subcase 3: 19 points homography
for i in range(0,730):
    vector = np.dot(h_19points, np.transpose([x_line[i], y_line[i], 1]))
    x_img = vector[0]
    y_img = vector[1]
    w = vector[2]
    x_new = int(x_img / w)
    y_new = int(y_img / w)
    x_line_homography_19points.append(x_new)
    y_line_homography_19points.append(y_new)
    cv2.circle(img_19points, (x_new, y_new), 4, (0, 255, 0), -1)
    cv2.imshow('image_19points', img_19points)
    # Press 'q' to close the image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Computation of the mean square error:
#   4 points  --> 11 points (MSE_4_11_blue)
#   11 points --> 19 points (MSE_11_19_blue)
#   19 points --> 4 points  (MSE_19_4_blue)

MSE_4_11_x=0
MSE_11_19_x=0
MSE_19_4_x=0
MSE_4_11_y=0
MSE_11_19_y=0
MSE_19_4_y=0

for i in range(0,730):
    MSE_4_11_x=MSE_4_11_x+(x_line_homography_4points[i]-x_line_homography_11points[i])**2
    MSE_4_11_y = MSE_4_11_y + (y_line_homography_4points[i] - y_line_homography_11points[i])**2
    MSE_11_19_x=MSE_11_19_x+(x_line_homography_11points[i]-x_line_homography_19points[i])**2
    MSE_11_19_y = MSE_4_11_y + (y_line_homography_11points[i] - y_line_homography_19points[i])**2
    MSE_19_4_x=MSE_19_4_x+(x_line_homography_19points[i]-x_line_homography_4points[i])**2
    MSE_19_4_y = MSE_19_4_y + (y_line_homography_19points[i] - y_line_homography_4points[i])**2

MSE_4_11_x=MSE_4_11_x/len(x_line_homography_4points)
MSE_4_11_y=MSE_4_11_y/len(y_line_homography_4points)
MSE_11_19_x=MSE_11_19_x/len(x_line_homography_11points)
MSE_11_19_y=MSE_11_19_y/len(y_line_homography_11points)
MSE_19_4_x=MSE_19_4_x/len(x_line_homography_19points)
MSE_19_4_y=MSE_19_4_y/len(y_line_homography_19points)

# Write on a file the results
MSE_file.write("[GREEN] Mean Square Error homography 4 points --> homography 11 points: (")
MSE_file.write(str(round(MSE_4_11_x,2)))
MSE_file.write(",")
MSE_file.write(str(round(MSE_4_11_y,2)))
MSE_file.write(")\n")
MSE_file.write("[GREEN] Mean Square Error homography 11 points --> homography 19 points: (")
MSE_file.write(str(round(MSE_11_19_x,2)))
MSE_file.write(",")
MSE_file.write(str(round(MSE_11_19_y,2)))
MSE_file.write(")\n")
MSE_file.write("[GREEN] Mean Square Error homography 19 points --> homography 4 points: (")
MSE_file.write(str(round(MSE_19_4_x,2)))
MSE_file.write(",")
MSE_file.write(str(round(MSE_19_4_y,2)))
MSE_file.write(")\n")

# Case 3: red line
x_line=np.arange(255,1131)
y_line=np.full(876,450,int)
# Initialization of the lists for the points
x_line_homography_4points=list()
y_line_homography_4points=list()
x_line_homography_11points=list()
y_line_homography_11points=list()
x_line_homography_19points=list()
y_line_homography_19points=list()

# Compute the homography of the line
# Subcase 1: 4 points homography
for i in range(0,875):
    vector = np.dot(h_4points, np.transpose([x_line[i], y_line[i], 1]))
    x_img = vector[0]
    y_img = vector[1]
    w = vector[2]
    x_new = int(x_img / w)
    y_new = int(y_img / w)
    x_line_homography_4points.append(x_new)
    y_line_homography_4points.append(y_new)
    cv2.circle(img_4points, (x_new, y_new), 4, (0, 0, 255), -1)
    cv2.imshow('image_4points', img_4points)
    # Press 'q' to close the image and then choose another point on the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Subcase 2: 11 points homography
for i in range(0,875):
    vector = np.dot(h_11points, np.transpose([x_line[i], y_line[i], 1]))
    x_img = vector[0]
    y_img = vector[1]
    w = vector[2]
    x_new = int(x_img / w)
    y_new = int(y_img / w)
    x_line_homography_11points.append(x_new)
    y_line_homography_11points.append(y_new)
    cv2.circle(img_11points, (x_new, y_new), 4, (0, 0, 255), -1)
    cv2.imshow('image_11points', img_11points)
    # Press 'q' to close the image and then choose another point on the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Subcase 3: 19 points homography
for i in range(0,875):
    vector = np.dot(h_19points, np.transpose([x_line[i], y_line[i], 1]))
    x_img = vector[0]
    y_img = vector[1]
    w = vector[2]
    x_new = int(x_img / w)
    y_new = int(y_img / w)
    x_line_homography_19points.append(x_new)
    y_line_homography_19points.append(y_new)
    cv2.circle(img_19points, (x_new, y_new), 4, (0, 0, 255), -1)
    cv2.imshow('image_19points', img_19points)
    # Press 'q' to close the image and then choose another point on the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Computation of the mean square error:
#   4 points  --> 11 points (MSE_4_11_blue)
#   11 points --> 19 points (MSE_11_19_blue)
#   19 points --> 4 points  (MSE_19_4_blue)

MSE_4_11_x=0
MSE_11_19_x=0
MSE_19_4_x=0
MSE_4_11_y=0
MSE_11_19_y=0
MSE_19_4_y=0

for i in range(0,875):
    MSE_4_11_x=MSE_4_11_x+(x_line_homography_4points[i]-x_line_homography_11points[i])**2
    MSE_4_11_y = MSE_4_11_y + (y_line_homography_4points[i] - y_line_homography_11points[i])**2
    MSE_11_19_x=MSE_11_19_x+(x_line_homography_11points[i]-x_line_homography_19points[i])**2
    MSE_11_19_y = MSE_4_11_y + (y_line_homography_11points[i] - y_line_homography_19points[i])**2
    MSE_19_4_x=MSE_19_4_x+(x_line_homography_19points[i]-x_line_homography_4points[i])**2
    MSE_19_4_y = MSE_19_4_y + (y_line_homography_19points[i] - y_line_homography_4points[i])**2

MSE_4_11_x=MSE_4_11_x/len(x_line_homography_4points)
MSE_4_11_y=MSE_4_11_y/len(y_line_homography_4points)
MSE_11_19_x=MSE_11_19_x/len(x_line_homography_11points)
MSE_11_19_y=MSE_11_19_y/len(y_line_homography_11points)
MSE_19_4_x=MSE_19_4_x/len(x_line_homography_19points)
MSE_19_4_y=MSE_19_4_y/len(y_line_homography_19points)

# Write on a file the results
MSE_file.write("[RED] Mean Square Error homography 4 points --> homography 11 points: (")
MSE_file.write(str(round(MSE_4_11_x,2)))
MSE_file.write(",")
MSE_file.write(str(round(MSE_4_11_y,2)))
MSE_file.write(")\n")
MSE_file.write("[RED] Mean Square Error homography 11 points --> homography 19 points: (")
MSE_file.write(str(round(MSE_11_19_x,2)))
MSE_file.write(",")
MSE_file.write(str(round(MSE_11_19_y,2)))
MSE_file.write(")\n")
MSE_file.write("[RED] Mean Square Error homography 19 points --> homography 4 points: (")
MSE_file.write(str(round(MSE_19_4_x,2)))
MSE_file.write(",")
MSE_file.write(str(round(MSE_19_4_y,2)))
MSE_file.write(")\n")

# Save images
cv2.imwrite('../Output/Accuracy/Video/Horizontal_lines.png', smallFrame)
cv2.imwrite('../Output/Accuracy/Homography/Horizontal_lines_4points.png', img_4points)
cv2.imwrite('../Output/Accuracy/Homography/Horizontal_lines_11points.png', img_11points)
cv2.imwrite('../Output/Accuracy/Homography/Horizontal_lines_19points.png', img_19points)

# When everything done, release the video capture object
cap.release()

# Close all the frames
cv2.destroyAllWindows()

# Close file
MSE_file.close()
