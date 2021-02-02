"""
    This script compute the trajectory of a multiple players and reproduce the trajectory on the basketball diagram.
    Moreover, it evalueates the length of the trajectory, the acceleration and the average speed of the player in a given timestep.
"""
import cv2 as cv
import numpy as np
import yaml
import scipy as sp
from random import randint
import webcolors
#import imutils
from sklearn.metrics import mean_squared_error
#import sys
import matplotlib.pyplot as plt
#import seaborn as sns
import time


trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker

def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

# Instantiate OCV kalman filter
class KalmanFilter:
    def __init__(self, index = 0):
        self.kf = cv.KalmanFilter(4, 2, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.index = index

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted

# Control version of opencv library
(major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')

# Read homography matrix
with open('homography_19points.yaml') as f:
    loadeddict = yaml.full_load(f)

hloaded = loadeddict.get('homography')
h = np.asarray(hloaded)
# Read the image
img = cv.imread('../Sources/Map/basket_field.jpg')
# Set output video
fourcc = cv.VideoWriter_fourcc(*'DIVX')
output1 = '../Output/Tracking/tracked_players.avi'
output2 = '../Output/Tracking/tracked_homography.avi'
output3 = "../Output/Tracking/bounding_box.png"
output4 = "../Output/Tracking/data_players.txt"
output5 = "../Output/Tracking/results.png"
tau = 0.6
out = cv.VideoWriter(output1, fourcc, 25.0, (1344, 756))
points = cv.VideoWriter(output2, fourcc, 25.0, (1081, 612))
cap = cv.VideoCapture('../Output/Video/clip3.mp4')
fps = cap.get(cv.CAP_PROP_FPS)
cap.isOpened()
ok, frame = cap.read()
smallFrame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
kalman_filters = []
kalman_filtersp1 = []
kalman_filtersp2 = []
bboxes = []
colors = []
histo = []

while True:
    # draw bounding boxes over objects
    # selectROI's default behaviour is to draw box starting from the center
    # when fromCenter is set to false, you can draw box starting from top left corner
    bbox = cv.selectROI("ROI", smallFrame, False)
    crop_img = smallFrame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
    hist_1, _ = np.histogram(crop_img, bins=256, range=[0, 255])
    histo.append(hist_1)
    bboxes.append(bbox)
    rgb = (randint(0, 255), randint(0, 255), randint(0, 255))
    actual_name, closest_name = get_colour_name(rgb)
    rgb = webcolors.name_to_rgb(closest_name)
    colors.append((rgb[0], rgb[1], rgb[2]))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv.waitKey(0) & 0xFF
    if (k == 113):  # q is pressed
        break
cv.destroyWindow("ROI")
print('Selected bounding boxes {}'.format(bboxes))
multiTracker = cv.legacy.MultiTracker_create()
# List for saving points of tracking in the basketball diagram (homography)
x_sequence_image = list()
y_sequence_image = list()
x_sequences = list()
y_sequences = list()
#ok, frame = cap.read()
#smallFrame = cv.resize(frame, (0, 0), fx=0.35, fy=0.35)
i = 0
trackerType = "CSRT"
for bbox in bboxes:
  multiTracker.add(createTrackerByName(trackerType), smallFrame, bbox)
  x_sequences.append(list())
  y_sequences.append(list())
  kalman_filters.append(KalmanFilter())
  kalman_filtersp1.append(KalmanFilter())
  kalman_filtersp2.append(KalmanFilter())
  i = i + 1

i = 0
for bbox in bboxes:
    tracking_point = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3]))
    cv.circle(smallFrame, (tracking_point[0], tracking_point[1]), 4, (255, 200, 0), -1)
    cv.rectangle(smallFrame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
    # Compute the point in the homographed space: destination point(image)=homography matrix*source point(video)
    vector = np.dot(h, np.transpose([tracking_point[0], tracking_point[1], 1]))
    # Evaluation of the vector
    tracking_point_img = (vector[0], vector[1])
    w = vector[2]
    tracking_point_new = (int(tracking_point_img[0] / w), int(tracking_point_img[1] / w))
    x_sequences[i].append(tracking_point_new[0])
    y_sequences[i].append(tracking_point_new[1])
    cv.circle(img, (tracking_point_new[0], tracking_point_new[1]), 4, colors[i], -1)
    i = i + 1

# Save and visualize the chosen bounding box and its point used for homography
cv.imwrite(output3, smallFrame)
cv.putText(smallFrame,"Selected Bounding Boxes. PRESS SPACE TO CONTINUE...", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
cv.imshow("bounding box", smallFrame)
cv.waitKey(0)
cv.destroyWindow("bounding box")

indice = 1
start = time.time()
# Loop for tracking
while (1):
    if indice <= 1 | indice >= 50:
        ok, frame = cap.read()
    if ok:
        # Resize the dimension of the frame
        smallFrame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
        cv.putText(smallFrame, trackerType + " Tracker", (100, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        ok, boxes = multiTracker.update(smallFrame)
        # Update position of the bounding box
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            # Computation of the new position of the tracking point
            tracking_point = (int(newbox[0] + newbox[2] / 2), int(newbox[1] + newbox[3]))
            predictedCoords = kalman_filters[i].Estimate(tracking_point[0], tracking_point[1])
            p1 = kalman_filtersp1[i].Estimate(p1[0], p1[1])
            p2 = kalman_filtersp2[i].Estimate(p2[0], p2[1])
            # Compute the point in the homographed space: destination point(image)=homography matrix*source point(video)
            vector = np.dot(h, np.transpose([predictedCoords[0][0], predictedCoords[1][0], 1]))
            # Evaluation of the vector
            if indice <= 50:
                indice = indice+1
            else:
                tracking_point_img = (vector[0], vector[1])
                w = vector[2]
                tracking_point_new = (int(tracking_point_img[0] / w), int(tracking_point_img[1] / w))
                # Add new position to list of points for the homographed space
                x_sequences[i].append(tracking_point_new[0])
                y_sequences[i].append(tracking_point_new[1])
                # computation of the predicted bounding box
                punto1 = (int(p1[0]), int(p1[1]))
                punto2 = (int(p2[0]), int(p2[1]))
                bbox_new = (punto1[0], punto1[1], punto2[0]-punto1[0], punto2[1]-punto1[1])
                crop_img = smallFrame[int(bbox_new[1]):int(bbox_new[1] + bbox_new[3]), int(bbox_new[0]):int(bbox_new[0] + bbox_new[2])]
                hist_2, _ = np.histogram(crop_img, bins=256, range=[0, 255])
                intersection = return_intersection(histo[i], hist_2)
                if intersection < tau:
                    print(intersection)
                    print("RE-INITIALIZE TRACKER CSRT n°%d" % i)
                    multiTracker = cv.legacy.MultiTracker_create()
                    for n, nb in enumerate(boxes):
                        boxi = (int(nb[0]), int(nb[1]), int(nb[2]), int(nb[3]))
                        if n == i:
                            multiTracker.add(createTrackerByName(trackerType), smallFrame, bbox_new)
                        else:
                            multiTracker.add(createTrackerByName(trackerType), smallFrame, boxi)

                    histo[i] = hist_2

                cv.rectangle(smallFrame, punto1, punto2, colors[i], 2, 1)
                cv.circle(smallFrame, (int(predictedCoords[0][0]), int(predictedCoords[1][0])), 4, colors[i], -1)
                cv.circle(img, (tracking_point_new[0], tracking_point_new[1]), 4, colors[i], -1)
                points.write(img)  # Save video for position tracking on the basketball diagram
                cv.imshow("Tracking-Homography", img)
                cv.imshow("Tracking", smallFrame)
        out.write(smallFrame)  # Save video frame by frame

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        # Tracking failure
        cv.putText(smallFrame, "Tracking failure detected", (100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
        break
cv.waitKey(0)
cv.destroyAllWindows()
end = time.time()

print(end - start)
# Post-processing
# 1) Apply a median filter to the two sequence of x, y coordinates in order to achieve a smoother trajectory
x_sequence_image = sp.signal.medfilt(x_sequence_image, 25)  # Window width of the filter MUST be ODD
y_sequence_image = sp.signal.medfilt(y_sequence_image, 25)
i = 0
position_x=list()
position_y=list()
for bbox in bboxes:
    x_sequences[i] = sp.signal.medfilt(x_sequences[i], 25)
    y_sequences[i] = sp.signal.medfilt(y_sequences[i], 25)
    # Draw the trajectory on the basketball diagram
    pts = np.column_stack((x_sequences[i], y_sequences[i]))
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv.polylines(img, [pts], False, colors[i], 2)
    position_x.append(list())
    position_y.append(list())
    i = i + 1

# Show the result
cv.imshow("Smoothing", img)
cv.waitKey(0)
cv.destroyWindow("Smoothing")

# Evaluation of the shift, acceleration and the average speed of the players in real world coordinates
# Step 1: Compute the position of the smoothed vector of position in real world coordinates
# Step 2: Evaluate the length of the trajectory using an Euclidian distance between 2 successive points and sum them together
#         and compute the acceleration values
# Step 3: Compute the velocity and the total length of the trajectory

# Step 1
flag = 0
indice = 0
for bbox in bboxes:
    x_sequence_image = x_sequences[indice]
    y_sequence_image = y_sequences[indice]
    for i in range(0, len(x_sequence_image) - 1):
        # x coordinate
        length = x_sequence_image[i] - 38
        proportion = length / 1008.0
        position_x[indice].append(28 * proportion)
        # y coordinate
        length = y_sequence_image[i] - 28
        proportion = length / 545.0
        position_y[indice].append(15 * proportion)

    indice = indice+1
# Step 2
shift = 0
px = list()
py = list()
indice = 0
const = 0
f= open(output4,"w+")
f.write("TIME CONSUMED FOR TRACKING: %f\r\n" % (end - start))
for bbox in bboxes:
    px = position_x[indice]
    py = position_y[indice]
    shift = 0
    rgb = colors[indice]
    actual_name, closest_name = get_colour_name(rgb)
    f.write("\n\n")
    f.write("TRACKER COLOR %s\r\n" % closest_name)
    f.write("ACCELERATION:\r\n")
    iter_frame = 1
    for i in range(0, len(px) - 1):
        #compute here the accelleration for space sample
        shift = shift + np.sqrt((px[i + 1] - px[i]) ** 2 + (py[i + 1] - py[i]) ** 2) #steve updated from math.sqrt to np.sqrt
        if i == 50*iter_frame:
            if iter_frame == 1:
                shift_prec = shift
                speed1 = shift_prec / 2
                average_acceleration1 = speed1 / 2
                f.write("Detection done in the first 2 seconds\r\n")
                f.write("route space:%f, time step 2 sec\r\n" % shift_prec)
                f.write("acceleration: %f\r\n" % average_acceleration1)
            else:
                t1 = (((2*fps)*iter_frame)-(2*fps)) / fps
                t2 = ((2*fps)*iter_frame) / fps
                speed2 = (shift-shift_prec) / 2
                average_acceleration2 = speed2 / 2 - average_acceleration1
                average_acceleration1 = average_acceleration2
                f.write("Detection done in the time sample %d - %d sec\r\n" % (t2, t1))
                f.write("route space:%f, time step 2 sec\r\n" % (shift-shift_prec))
                f.write("acceleration: %f\r\n" % average_acceleration2)
                shift_prec = shift

            iter_frame = iter_frame + 1
            f.write("\n")
# Step 3
    # Print of the results
    # Evaluation of the average speed: speed=space/time
    average_speed = shift / (len(px)/fps)
    string1 = "trajectory length "
    string2 = str(round(shift, 2))
    string3 = "[m]"
    string = string1 + string2 + string3
    f.write("%s\r\n\n" % string)
    string1 = "average speed "
    string2 = str(round(average_speed, 2))
    string3 = "[m/s]"
    string = string1 + string2 + string3
    f.write("%s\r\n" % string)
    const=350+const
    cv.imwrite(output5, img)
    cv.imshow("Result", img)
    #cv.waitKey(0)
    indice = indice + 1
cap.release()
cv.destroyAllWindows()
f.close()
