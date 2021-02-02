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

# Function for point selection
def mouseCallback(event,x,y,flags,param):
    global ix
    global iy

    if event == cv2.EVENT_LBUTTONDOWN:
        # saves the position of the last click
        ix = x # x coordinate
        iy = y # y coordinate

# Function for draw a circle given the point coordinates
def draw_circle_onscreen(frame, x,y):
    cv2.circle(frame, (x,y), 4,(255, 0, 0),-1)

# Open a window
cv2.namedWindow('frame')
# Mouse callback has to be set only once
cv2.setMouseCallback('frame',mouseCallback) # mouse callback has to be set only once

# Open the video
cap = cv2.VideoCapture('../Output/Video/undistorted.primo_tempo.asf')
# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read the image
img = cv2.imread('../Sources/Map/basket_field.jpg')

# Read homography matrix
with open('configs/homography_19points.yaml') as f:
    loadeddict = yaml.load(f)

hloaded = loadeddict.get('homography')
h=np.asarray(hloaded)

while cap.isOpened():
    ret, frame = cap.read()
    # Reduce dimension of frame in order to show it on the screen
    smallFrame = cv2.resize(frame, (0, 0), fy=0.35, fx=0.35)
    # Draw circle on screen
    draw_circle_onscreen(smallFrame,ix,iy)
    # Compute the destination point: destination point(image)=homography matrix*source point(video)
    vector=np.dot(h,np.transpose([ix,iy,1]))
    # Evaluation of the vector
    x_img=vector[0]
    y_img=vector[1]
    w=vector[2]
    x_new=int(x_img/w)
    y_new=int(y_img/w)
    # Print the point on the image
    if ix != -1 & iy != -1:
        # You need to create a copy of the image if you want to visualize on the image only the current selected point
        new_img=copy.copy(img)
        cv2.circle(new_img, (x_new, y_new), 4, (0, 0, 255), -1)
        # Estimation of the point in real coordinates:
        #   1) coordinates in meters for points inside the basketball field
        #   2) for points on the board of the field, it will appear the word 'sideline'
        #   3) for points outiside the basketball field it will appear 'Point outiside the basketball Field'
        if x_new >= 38 and x_new <= 1046 and y_new >= 30 and y_new <= 573:
            # You can use string concatenation to generate the coordinates in meters to print
            string0="("
            length=x_new-38
            proportion=length/1008.0
            position_x=28*proportion
            string1 = str(round(position_x, 2))
            string2="m,"
            length=y_new-28
            proportion=length/545.0
            position_y=15*proportion
            string3 = str(round(position_y, 2))
            string4="m)"
            string=string0+string1+string2+string3+string4
            # You have to adjust the position of the printing on the sides of the image
            if x_new<80:
                cv2.putText(new_img, string, (x_new - 40 , y_new - 10), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            elif x_new>970:
                cv2.putText(new_img, string, (x_new - 175, y_new - 10), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(new_img,string,(x_new- 100, y_new- 10), font, 0.75,(0,0,255),2,cv2.LINE_AA)
        elif x_new>=0 and x_new<=1081 and y_new>=0 and y_new<=612:
            if y_new<10 and x_new>970:
                cv2.putText(new_img, 'sideline', (x_new - 50, y_new + 20), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            elif y_new<10 and x_new<980:
                cv2.putText(new_img, 'sideline', (x_new + 10, y_new + 20), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            elif y_new>590 and x_new<980:
                cv2.putText(new_img, 'sideline', (x_new - 50, y_new - 10), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            elif y_new>590 and x_new>980:
                cv2.putText(new_img, 'sideline', (x_new - 50, y_new - 10), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            elif y_new>=10 and y_new<=590 and x_new>980:
                cv2.putText(new_img, 'sideline', (x_new - 100, y_new + 10), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(new_img, 'sideline', (x_new + 10, y_new + 10), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(new_img, 'point outside the Basketball Court', (25, 600), font, 1, (50, 50, 50), 2, cv2.LINE_AA)
        cv2.imshow('image', new_img)
        # Press 'q' to close the image and then choose another point on the video
        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyWindow('image')
            ix, iy = -1,-1
    cv2.imshow('frame',smallFrame)
    # Press esc to close the video
    if cv2.waitKey(5) & 0xFF == 27:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

"""
    NOTE: 28=width of the basketball court in meters
          15=height of the basketball court in meters
"""