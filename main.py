'''
    This script compute the trajectory of a multiple players and reproduce the trajectory on the basketball diagram.
    Moreover, it evalueates the length of the trajectory, the acceleration and the average speed of the player in a given timestep.
'''
import cv2 as cv
import numpy as np
import yaml
import scipy as sp
from scipy import signal
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
import colorutils
from kalman_filter import KalmanFilter
from maskers import getMaskerByName


def createTracker(trackerType):
    trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    if trackerType == trackerTypes[0]:
        tracker = cv.legacy.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv.legacy.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv.legacy.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv.legacy.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv.legacy.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv.legacy.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv.legacy.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print(f'Incorrect tracker name, available trackers are: {trackerTypes}')
    return tracker


def drawPolyROI(event, x, y, flags, params):
    # :mouse callback function
    img2 = params["image"].copy()

    if event == cv.EVENT_LBUTTONDOWN:  # Left click, select point
        pts.append((x, y))
    if event == cv.EVENT_RBUTTONDOWN:  # Right click to cancel the last selected point
        pts.pop()
    if event == cv.EVENT_MBUTTONDOWN:  # Central button to display the polygonal mask
        mask = np.zeros(img2.shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))
        mask = cv.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv.fillPoly(mask.copy(), [points], (255, 255, 255))  # for ROI
        # Mask3 = cv.fillPoly(mask.copy(), [points], (0, 255, 0)) # for displaying images on the desktop

        show_image = cv.addWeighted(src1=img2, alpha=params["alpha"], src2=mask2, beta=1-params["alpha"], gamma=0)
        cv.putText(show_image, 'PRESS SPACE TO CONTINUE THE SELECTION...', (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv.imshow("ROI inspection", show_image)
        cv.waitKey(0)
        cv.destroyWindow("ROI inspection")
    if len(pts) > 0:  # Draw the last point in pts
        cv.circle(img2, pts[-1], 3, (0, 0, 255), -1)
    if len(pts) > 1:
        for i in range(len(pts) - 1):
            cv.circle(img2, pts[i], 4, (0, 0, 255), -1)  # x ,y is the coordinates of the mouse click place
            cv.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=1)
    cv.imshow('ROI', img2)


def returnIntersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection


#    _____ _   _ _____ _______
#   |_   _| \ | |_   _|__   __|
#     | | |  \| | | |    | |
#     | | | . ` | | |    | |
#    _| |_| |\  |_| |_   | |
#   |_____|_| \_|_____|  |_|


SHOW_MASKS = False
SHOW_HOMOGRAPHY = False
MANUAL_ROI_SELECTION = True
POLYNOMIAL_ROI = False

# Read congigurations
with open('config.yaml') as f:
    loadeddict = yaml.full_load(f)
    TRACKER = loadeddict.get('tracker')
    TAU = loadeddict.get('tau')
    RESIZE_FACTOR = loadeddict.get('resize_factor')

# Read homography matrix
with open('configs/homography_19points.yaml') as f:
    dict_homo = yaml.full_load(f)
    h = np.array(dict_homo.get('homography'))

img = cv.imread(loadeddict.get('input_image_homography'))


# Set input video
cap = cv.VideoCapture(loadeddict.get('input_video'))
fps = cap.get(cv.CAP_PROP_FPS)
if not cap.isOpened():
    exit("Input video not opened correctly")
ok, frame = cap.read()
smallFrame = cv.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
kalman_filters, kalman_filtersp1, kalman_filtersp2 = [], [], []
maskers = []
color_names_used = set()
bboxes = []
poly_roi = []
colors = []
histo = []

# Set output video
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter(loadeddict.get('out_players'), fourcc, 25.0, smallFrame.shape[1::-1])
out_mask = cv.VideoWriter(loadeddict.get('out_players_mask'), fourcc, 25.0, smallFrame.shape[1::-1])
points = cv.VideoWriter(loadeddict.get('out_homography'), fourcc, 25.0, img.shape[1::-1])


#    __  __          _____ _   _
#   |  \/  |   /\   |_   _| \ | |
#   | \  / |  /  \    | | |  \| |
#   | |\/| | / /\ \   | | | . ` |
#   | |  | |/ ____ \ _| |_| |\  |
#   |_|  |_/_/    \_\_____|_| \_|


if MANUAL_ROI_SELECTION:
    bbox = None
    if POLYNOMIAL_ROI:
        pts = []
        cv.namedWindow('ROI')
        cv.setMouseCallback('ROI', drawPolyROI, {"image": smallFrame, "alpha": 0.6})
        print("[INFO] Click the left button: select the point, right click: delete the last selected point, click the middle button: inspect the ROI area")
        print("[INFO] Press ‘S’ to determine the selection area and save it")
        print("[INFO] Press q or ESC to quit")
    while True:
        if POLYNOMIAL_ROI:
            key = cv.waitKey(1) & 0xFF
            #if key == ord('q') or key == 27:
            #    cv.destroyWindow('ROI')
            #    break
            if key == ord("s"): 
                print("[INFO] ROI coordinates:", pts)
                if len(pts) >= 3:
                    #self.poly_roi.append(pts[0])
                    poly_roi.append(pts)
                    bbox = cv.boundingRect(np.array(pts)) #extract the minimal Rectangular that fit the polygon just selected. This because Tracking algos work with rect. bbox
                    pts = []
                else:
                    print("Not enough points selected")  
        else:
            bbox = cv.selectROI('ROI', smallFrame, False)
            if bbox == (0, 0, 0, 0):  # no box selected
                cv.destroyWindow('ROI')
                break
            print('[INFO] Press q to quit selecting boxes and start tracking, or any other key to select next object')
           
        if bbox: #because the callback of the mouse doesn not block the main thread
            crop_img = smallFrame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
            hist_1, _ = np.histogram(crop_img, bins=256, range=[0, 255])
            histo.append(hist_1)
            bboxes.append(bbox)
            colors.append(colorutils.pickNewColor(color_names_used))    
            bbox = None
        else:
            time.sleep(0.2)
            
        if (cv.waitKey(0) & 0xFF == ord('q')):  # q is pressed
            cv.destroyWindow('ROI')
            break
else:
    """
    Example bounding boxes for clip3.mp4
    For RESIZE_FACTOR=0.25 -> [(205, 280, 22, 42), (543, 236, 17, 38), (262, 270, 16, 33), (722, 264, 21, 47)]
    For RESIZE_FACTOR=0.35 -> [(1013, 371, 25, 60), (367, 376, 21, 49), (566, 386, 35, 63)]
    """
    for bbox in [(725, 495, 53, 82)]:
        crop_img = smallFrame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        hist_1, _ = np.histogram(crop_img, bins=256, range=[0, 255])
        histo.append(hist_1)
        bboxes.append(bbox)
        colors.append(colorutils.pickNewColor(color_names_used))

print('Selected bounding boxes: {}'.format(bboxes))
multiTracker = cv.legacy.MultiTracker_create()

# List for saving points of tracking in the basketball diagram (homography)
x_sequence_image, y_sequence_image = [], []
x_sequences, y_sequences = [], []
#ok, frame = cap.read()
#smallFrame = cv.resize(frame, (0, 0), fx=0.35, fy=0.35)
for i, bbox in enumerate(bboxes):
    multiTracker.add(createTracker(TRACKER), smallFrame, bbox)
    x_sequences.append([])
    y_sequences.append([])

    kalman_filters.append(KalmanFilter())
    kalman_filtersp1.append(KalmanFilter())
    kalman_filtersp2.append(KalmanFilter())

    poly_roi_tmp = poly_roi[i] if POLYNOMIAL_ROI else None
    maskers.append(getMaskerByName(loadeddict.get('masker'), debug=True, frame=smallFrame, bbox=bbox, poly_roi=poly_roi_tmp))

    tracking_point = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3]))
    cv.circle(smallFrame, tracking_point, 4, (255, 200, 0), -1)
    cv.rectangle(smallFrame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
    # Compute the point in the homographed space: destination point(image)=homography matrix*source point(video)
    vector = np.dot(h, np.transpose([tracking_point[0], tracking_point[1], 1]))
    # Evaluation of the vector
    tracking_point_img = (vector[0], vector[1])
    w = vector[2]
    tracking_point_new = (int(tracking_point_img[0] / w), int(tracking_point_img[1] / w))
    x_sequences[i].append(tracking_point_new[0])
    y_sequences[i].append(tracking_point_new[1])
    cv.circle(img, tracking_point_new, 4, colors[i], -1)

# Save and visualize the chosen bounding box and its point used for homography
cv.imwrite(loadeddict.get('out_bboxes'), smallFrame)
cv.putText(smallFrame, 'Selected Bounding Boxes. PRESS SPACE TO CONTINUE...', (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
cv.imshow('Tracking', smallFrame)
cv.waitKey(0)


start = time.time()
index = 1

previousFrame = smallFrame
previousBoxes = bboxes
while (1):
    if index > 50:
        ok, frame = cap.read()
    if ok:
        smallFrame = cv.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        maskedFrame = np.zeros(smallFrame.shape, dtype=np.uint8)

        ok, boxes = multiTracker.update(smallFrame)

        # Update position of the bounding box
        for i, newbox in enumerate(boxes):
            p1_t = (int(newbox[0]), int(newbox[1]))
            p2_t = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))

            # Computation of the new position of the tracking point
            tracking_point = (int(newbox[0] + newbox[2] / 2), int(newbox[1] + newbox[3]))
            predictedCoords = kalman_filters[i].estimate(tracking_point[0], tracking_point[1])
            p1_k = kalman_filtersp1[i].estimate(p1_t[0], p1_t[1])
            p2_k = kalman_filtersp2[i].estimate(p2_t[0], p2_t[1])
            # Compute the point in the homographed space: destination point(image)=homography matrix*source point(video)
            vector = np.dot(h, np.transpose([predictedCoords[0][0], predictedCoords[1][0], 1]))
            if index <= 50:
                index += 1
            else:
                tracking_point_img = (vector[0], vector[1])
                w = vector[2]
                tracking_point_new = (int(tracking_point_img[0] / w), int(tracking_point_img[1] / w))
                # Add new position to list of points for the homographed space
                x_sequences[i].append(tracking_point_new[0])
                y_sequences[i].append(tracking_point_new[1])
                # computation of the predicted bounding box
                punto1_k = (int(p1_k[0]), int(p1_k[1]))
                punto2_k = (int(p2_k[0]), int(p2_k[1]))
                punto1_t = (int(p1_t[0]), int(p1_t[1]))
                punto2_t = (int(p2_t[0]), int(p2_t[1]))

                bbox_new = (int(punto1_k[0]), int(punto1_k[1]), int(punto2_k[0] - punto1_k[0]), int(punto2_k[1] - punto1_k[1]))
                bbox_new_t = (int(punto1_t[0]), int(punto1_t[1]), int(punto2_t[0] - punto1_t[0]), int(punto2_t[1] - punto1_t[1]))

                maskers[i].update(bbox=bbox_new_t, frame=smallFrame,
                                  prev_bbox=previousBoxes[i], prev_frame=previousFrame,
                                  point1_t=punto1_t, point2_t=punto2_t, point1_k=punto1_k, point2_k=punto1_k, color=colors[i])

                # RE-INITIALIZATION START
                crop_img = smallFrame[bbox_new[1]:bbox_new[1] + bbox_new[3], bbox_new[0]:bbox_new[0] + bbox_new[2]]
                hist_2, _ = np.histogram(crop_img, bins=256, range=[0, 255])
                intersection = returnIntersection(histo[i], hist_2)
                if intersection < TAU:
                    print('RE-INITIALIZE TRACKER CSRT n° %d' % i)
                    colors[i] = colorutils.pickNewColor(color_names_used)
                    multiTracker = cv.legacy.MultiTracker_create()
                    for n, nb in enumerate(boxes):
                        boxi = (nb[0], nb[1], nb[2], nb[3])
                        if n == i:
                            multiTracker.add(createTracker(TRACKER), smallFrame, bbox_new)
                        else:
                            multiTracker.add(createTracker(TRACKER), smallFrame, boxi)
                    histo[i] = hist_2
                # RE-INITIALIZATION END

                cv.putText(smallFrame, TRACKER + ' Tracker', (100, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
                cv.putText(smallFrame, '{:.2f}'.format(intersection), (punto1_k[0], punto1_k[1]-7), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv.circle(smallFrame, (int(predictedCoords[0][0]), int(predictedCoords[1][0])), 4, colors[i], -1)

                cv.circle(img, tracking_point_new, 4, colors[i], -1)
                points.write(img)  # Save video for position tracking on the basketball diagram

                # Compute masked frame
                maskedFrame[bbox_new[1]:bbox_new[1] + bbox_new[3], bbox_new[0]:bbox_new[0] + bbox_new[2]] = [255, 255, 255]

                # Show results
                cv.imshow('Tracking', smallFrame)
                if SHOW_MASKS:
                    cv.imshow('Tracking-Masks', maskedFrame)
                if SHOW_HOMOGRAPHY:
                    cv.imshow('Tracking-Homography', img)

        previousFrame = smallFrame.copy()
        previousBoxes = boxes.copy()

        if index > 50:
            out.write(smallFrame)  # Save video frame by frame
            out_mask.write(maskedFrame)  # Save masked video

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv.putText(smallFrame, 'Tracking failure detected', (100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
        break

cv.waitKey(0)
out.release()
out_mask.release()
points.release()
cv.destroyAllWindows()
end = time.time()
print(f'\nTotal time consumed for tracking: {(end - start):.2f}s')


#plt.hist([m.distances for m in maskers], bins=np.unique([m.distances for m in maskers]).size)
# plt.show()


#    _____          _   _____                             _
#   |  __ \        | | |  __ \                           (_)
#   | |__) |__  ___| |_| |__) | __ ___   ___ ___  ___ ___ _ _ __   __ _
#   |  ___/ _ \/ __| __|  ___/ '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
#   | |  | (_) \__ \ |_| |   | | | (_) | (_|  __/\__ \__ \ | | | | (_| |
#   |_|   \___/|___/\__|_|   |_|  \___/ \___\___||___/___/_|_| |_|\__, |
#                                                                  __/ |
#                                                                 |___/
# (Montibeller project)

# 1) Apply a median filter to the two sequence of x, y coordinates in order to achieve a smoother trajectory
x_sequence_image = sp.signal.medfilt(x_sequence_image, 25)  # Window width of the filter MUST be ODD
y_sequence_image = sp.signal.medfilt(y_sequence_image, 25)
position_x = []
position_y = []
for i, bbox in enumerate(bboxes):
    x_sequences[i] = sp.signal.medfilt(x_sequences[i], 25)
    y_sequences[i] = sp.signal.medfilt(y_sequences[i], 25)
    # Draw the trajectory on the basketball diagram
    pts = np.column_stack((x_sequences[i], y_sequences[i]))
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv.polylines(img, [pts], False, colors[i], 2)
    position_x.append([])
    position_y.append([])

# Show the result
if SHOW_HOMOGRAPHY:
    cv.imshow('Smoothing', img)
    cv.waitKey(0)
    cv.destroyWindow('Smoothing')

# Evaluation of the shift, acceleration and the average speed of the players in real world coordinates
# Step 1: Compute the position of the smoothed vector of position in real world coordinates
# Step 2: Evaluate the length of the trajectory using an Euclidian distance between 2 successive points and sum them together
#         and compute the acceleration values
# Step 3: Compute the velocity and the total length of the trajectory

# Step 1
flag = 0
index = 0
for bbox in bboxes:
    x_sequence_image = x_sequences[index]
    y_sequence_image = y_sequences[index]
    for i in range(0, len(x_sequence_image) - 1):
        # x coordinate
        length = x_sequence_image[i] - 38
        proportion = length / 1008.0
        position_x[index].append(28 * proportion)
        # y coordinate
        length = y_sequence_image[i] - 28
        proportion = length / 545.0
        position_y[index].append(15 * proportion)
    index += 1
# Step 2
shift = 0
index = 0
px, py = [], []
f = open(loadeddict.get('out_players_data'), 'w+')
f.write('TIME CONSUMED FOR TRACKING: %f\r\n' % (end - start))
for bbox in bboxes:
    px = position_x[index]
    py = position_y[index]
    shift = 0
    rgb = colors[index]
    actual_name, closest_name = colorutils.getColorName(rgb)
    f.write('\n\n')
    f.write('TRACKER COLOR %s\r\n' % closest_name)
    f.write('ACCELERATION:\r\n')
    iter_frame = 1
    shift_prec, average_acceleration1 = None, None
    for i in range(0, len(px) - 1):
        # compute here the accelleration for space sample
        shift = shift + np.sqrt((px[i + 1] - px[i]) ** 2 + (py[i + 1] - py[i]) ** 2)  # steve updated from math.sqrt to np.sqrt
        if i == 50*iter_frame:
            if iter_frame == 1:
                shift_prec = shift
                speed1 = shift_prec / 2
                average_acceleration1 = speed1 / 2
                f.write('Detection done in the first 2 seconds\r\n')
                f.write('route space:%f, time step 2 sec\r\n' % shift_prec)
                f.write('acceleration: %f\r\n' % average_acceleration1)
            else:
                t1 = (((2 * fps) * iter_frame) - (2 * fps)) / fps
                t2 = ((2 * fps) * iter_frame) / fps
                speed2 = (shift - shift_prec) / 2
                average_acceleration2 = speed2 / 2 - average_acceleration1
                average_acceleration1 = average_acceleration2
                f.write('Detection done in the time sample %d - %d sec\r\n' % (t2, t1))
                f.write('route space:%f, time step 2 sec\r\n' % (shift - shift_prec))
                f.write('acceleration: %f\r\n' % average_acceleration2)
                shift_prec = shift

            iter_frame += 1
            f.write('\n')
# Step 3
    # Print of the results
    # Evaluation of the average speed: speed=space/time
    average_speed = shift / (len(px)/fps)
    f.write(f'trajectory length {shift:.2f}[m]\r\n\n')
    f.write(f'average speed {average_speed:.2f}[m/s]\r\n\n')
    cv.imwrite(loadeddict.get('out_tracking_results'), img)
    cv.imshow('Result', img)
    index += 1
cap.release()
cv.destroyAllWindows()
f.close()
