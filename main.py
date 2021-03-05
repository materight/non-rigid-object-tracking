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
from benchmark import computeBenchmark


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

selectingLineBg, selectingLineFg = False, False
def drawLineROI(event, x, y, flags, params):
    global selectingLineBg, selectingLineFg
    # :mouse callback function
    img2 = params["image"].copy()
    bgpts = params["bgpoints"]
    fgpts = params["fgpoints"]
    bgmask = params["bgmask"]
    fgmask = params["fgmask"]
    selection_history = params["selection_history"]
    if event == cv.EVENT_LBUTTONDOWN: 
        selectingLineBg = True
        bgpts.append([])
        selection_history.append('bg')
    elif event == cv.EVENT_LBUTTONUP:
        selectingLineBg = False
    if event == cv.EVENT_RBUTTONDOWN:  
        selectingLineFg = True
        fgpts.append([])
        selection_history.append('fg')
    elif event == cv.EVENT_RBUTTONUP:
        selectingLineFg = False
    elif event == cv.EVENT_MOUSEMOVE:
        if selectingLineBg: bgpts[-1].append((x, y))
        elif selectingLineFg: fgpts[-1].append((x, y))
    
    def drawPoints(img, pts, color):
        for seg in pts: # Draw points in pts
            for i, p in enumerate(seg):
                cv.circle(img, p, 1, color, -1)
                if i > 0: cv.line(img, seg[i], seg[i - 1], color=color, thickness=2)
    
    drawPoints(img2, bgpts, (0,0,255))
    drawPoints(img2, fgpts, (0,255,0))
    drawPoints(bgmask, bgpts, 255)
    drawPoints(fgmask, fgpts, 255)
    cv.imshow('ROI-lines', img2)


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

DEBUG = True
SHOW_MASKS = False
SHOW_HOMOGRAPHY = False
MANUAL_ROI_SELECTION = True
POLYNOMIAL_ROI = True

WINDOW_HEIGHT = 700


# Read congigurations
with open('config.yaml') as f:
    loadeddict = yaml.full_load(f)
    TRACKER = loadeddict.get('tracker')
    MASKER = loadeddict.get('masker')
    TAU = loadeddict.get('tau')
    RESIZE_FACTOR = loadeddict.get('resize_factor')

# Read homography matrix
with open('configs/homography_19points.yaml') as f:
    dict_homo = yaml.full_load(f)
    h = np.array(dict_homo.get('homography'))

img = cv.imread(loadeddict.get('input_image_homography'))


# Set input video
cap = cv.VideoCapture(loadeddict.get('input_video'))
ratio = cap.get(cv.CAP_PROP_FRAME_WIDTH) / cap.get(cv.CAP_PROP_FRAME_HEIGHT)
WINDOW_WIDTH = int(WINDOW_HEIGHT * ratio)
fps = cap.get(cv.CAP_PROP_FPS)
if not cap.isOpened():
    exit("Input video not opened correctly")
ok, frame = cap.read()
smallFrame = cv.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
first_frame = smallFrame.copy()
kalman_filters, kalman_filtersp1, kalman_filtersp2 = [], [], []
maskers = []
color_names_used = set()
bboxes = []
poly_roi = []
colors = []
histo = []

# Set output video
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter(loadeddict.get('out_players'), fourcc, fps, smallFrame.shape[1::-1])
out_mask = cv.VideoWriter(loadeddict.get('out_players_mask'), fourcc, fps, smallFrame.shape[1::-1])
points = cv.VideoWriter(loadeddict.get('out_homography'), fourcc, fps, img.shape[1::-1])


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
        print("[INFO] Press ENTER to determine the selection area and save it")
        print("[INFO] Press q or ESC to quit")
    while True:
        if POLYNOMIAL_ROI:
            #key = cv.waitKey(1) & 0xFF
            # if key == ord('q') or key == 27:
            #    cv.destroyWindow('ROI')
            #    break
            """if key == ord("s"):
                print("[INFO] ROI coordinates:", pts)
                if len(pts) >= 3:
                    #self.poly_roi.append(pts[0])
                    poly_roi.append(pts)
                    bbox = cv.boundingRect(np.array(pts)) #extract the minimal Rectangular that fit the polygon just selected. This because Tracking algos work with rect. bbox
                    pts = []
                else:
                    print("Not enough points selected")  """
        else:
            bbox = cv.selectROI('ROI', smallFrame, False)
            if bbox == (0, 0, 0, 0):  # no box selected
                cv.destroyWindow('ROI')
                break
            print('[INFO] Press q to quit selecting boxes and start tracking, or any other key to select next object')

        if bbox:  # because the callback of the mouse doesn not block the main thread
            crop_img = smallFrame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
            hist_1, _ = np.histogram(crop_img, bins=256, range=[0, 255])
            histo.append(hist_1)
            bboxes.append(bbox)
            colors.append(colorutils.pickNewColor(color_names_used))
            bbox = None
        else:
            time.sleep(0.2)

        key = cv.waitKey(0) & 0xFF
        if (key == ord('q')):  # q is pressed
            cv.destroyWindow('ROI')
            break
        if POLYNOMIAL_ROI and key == ord("\r"):
            print("[INFO] ROI coordinates:", pts)
            if len(pts) >= 3:
                # self.poly_roi.append(pts[0])
                poly_roi.append(pts)
                bbox = cv.boundingRect(np.array(pts))  # extract the minimal Rectangular that fit the polygon just selected. This because Tracking algos work with rect. bbox
                pts = []
            else:
                print("Not enough points selected")
else:
    """
    Example bounding boxes for clip3.mp4
    For RESIZE_FACTOR=0.25 -> [(205, 280, 22, 42), (543, 236, 17, 38), (262, 270, 16, 33), (722, 264, 21, 47)]
    For RESIZE_FACTOR=0.35 -> [(1013, 371, 25, 60), (367, 376, 21, 49), (566, 386, 35, 63)]

    Example poly_roi for clip3.mp4
    [(735, 499), (747, 512), (759, 528), (762, 547), (773, 569), (768, 572), (753, 551), (747, 575), (738, 575), (739, 533), (731, 527), (730, 504)]
    [(952, 443), (955, 452), (957, 467), (957, 483), (963, 494), (962, 504), (956, 507), (946, 492), (940, 507), (930, 505), (930, 495), (935, 483), (938, 469), (938, 458), (943, 440)]
    Example for dino.mp4
    [(187, 21), (193, 21), (198, 24), (200, 37), (207, 28), (213, 23), (222, 21), (232, 24), (237, 26), (239, 35), (237, 44), (239, 52), (246, 61), (253, 63), (247, 71), (235, 63), (229, 61), (223, 59), (220, 65), (217, 72), (214, 78), (209, 77), (207, 74), (212, 67), (210, 58), (204, 54), (198, 53), (190, 52), (191, 46), (190, 38), (190, 30), (184, 26)]
    Example for frog.mp4
    [(170, 67), (182, 60), (201, 62), (219, 65), (227, 64), (237, 66), (219, 87), (209, 100), (196, 111), (198, 98), (188, 82), (173, 76)]
    Example for soldier.mp4
    [(418, 16), (429, 22), (435, 35), (430, 45), (424, 51), (431, 65), (437, 79), (440, 97), (446, 119), (440, 123), (434, 113), (420, 114), (414, 114), (414, 127), (417, 148), (416, 167), (411, 187), (408, 198), (423, 203), (417, 209), (402, 212), (395, 205), (398, 189), (399, 177), (401, 165), (404, 158), (395, 156), (390, 168), (378, 176), (366, 185), (357, 196), (351, 209), (344, 196), (346, 179), (360, 169), (376, 160), (366, 146), (365, 133), (361, 116), (365, 105), (347, 101), (358, 84), (373, 79), (383, 61), (396, 48), (401, 37), (403, 26), (409, 19)]

    """
    if POLYNOMIAL_ROI:
        pts =  [(418, 16), (429, 22), (435, 35), (430, 45), (424, 51), (431, 65), (437, 79), (440, 97), (446, 119), (440, 123), (434, 113), (420, 114), (414, 114), (414, 127), (417, 148), (416, 167), (411, 187), (408, 198), (423, 203), (417, 209), (402, 212), (395, 205), (398, 189), (399, 177), (401, 165), (404, 158), (395, 156), (390, 168), (378, 176), (366, 185), (357, 196), (351, 209), (344, 196), (346, 179), (360, 169), (376, 160), (366, 146), (365, 133), (361, 116), (365, 105), (347, 101), (358, 84), (373, 79), (383, 61), (396, 48), (401, 37), (403, 26), (409, 19)] 
        for i, _ in enumerate(pts): pts[i] = (pts[i][0], pts[i][1])
        poly_roi.append(pts)
        bbox = cv.boundingRect(np.array(pts))
        example_bboxes = [bbox]
    else:
        example_bboxes = [(725, 495, 53, 82)]

    for bbox in example_bboxes:
        crop_img = smallFrame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        hist_1, _ = np.histogram(crop_img, bins=256, range=[0, 255])
        histo.append(hist_1)
        bboxes.append(bbox)
        colors.append(colorutils.pickNewColor(color_names_used))

# Select background and foreground points
bgmask = np.zeros(smallFrame.shape[:2], dtype=np.uint8)
fgmask = np.zeros(smallFrame.shape[:2], dtype=np.uint8)
if False: # MASKER == 'GrabCut':
    bgline, fgline, selection_history = [], [], []
    cv.namedWindow('ROI-lines')
    cv.imshow('ROI-lines', smallFrame)
    cv.setMouseCallback('ROI-lines', drawLineROI, {"image": smallFrame, "bgpoints": bgline, "fgpoints": fgline, "bgmask": bgmask, "fgmask": fgmask, "selection_history": selection_history})
    print("[INFO] Click left button: select background points, right button: select foreground points")
    print("[INFO] Press c to delete the last selected segment")
    print("[INFO] Press SPACE or ENTER to quit")
    while True:
        key = cv.waitKey(0) & 0xFF
        if key == ord('c') and len(selection_history) > 0:
            if selection_history[-1] == 'bg': bgline.pop()
            else: fgline.pop()
            selection_history.pop()
        if key == ord(' ') or key == ord("\r"):  # q or enter is pressed
            cv.destroyWindow('ROI-lines')
            break

print('Selected bounding boxes: {}'.format(bboxes))
multiTracker = cv.legacy.MultiTracker_create()

# List for saving points of tracking in the basketball diagram (homography)
x_sequence_image, y_sequence_image = [], []
x_sequences, y_sequences = [], []
for i, bbox in enumerate(bboxes):
    multiTracker.add(createTracker(TRACKER), smallFrame, bbox)
    x_sequences.append([])
    y_sequences.append([])

    kalman_filters.append(KalmanFilter())
    kalman_filtersp1.append(KalmanFilter())
    kalman_filtersp2.append(KalmanFilter())

    maskers.append(getMaskerByName(loadeddict.get('masker'),
                                   debug=DEBUG,
                                   frame=smallFrame,
                                   bbox=bbox,
                                   poly_roi=poly_roi[i] if POLYNOMIAL_ROI else None,
                                   bgmask=bgmask, 
                                   fgmask=fgmask,
                                   update_mask=loadeddict.get('update_mask')
                                   ))

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
cv.namedWindow('Tracking')
#cv.resizeWindow('Tracking', WINDOW_WIDTH,  WINDOW_HEIGHT)
cv.imshow('Tracking', smallFrame)
cv.waitKey(0)

if SHOW_MASKS:
    cv.namedWindow('Tracking-Masks', cv.WINDOW_NORMAL)
    cv.resizeWindow('Tracking-Masks', WINDOW_WIDTH,  WINDOW_HEIGHT)
if SHOW_HOMOGRAPHY:
    cv.namedWindow('Tracking-Homography', cv.WINDOW_NORMAL)
    cv.resizeWindow('Tracking-Homography', WINDOW_WIDTH,  WINDOW_HEIGHT)


benchmarkDist = []
start = time.time()
index = 0
cap = cv.VideoCapture(loadeddict.get('input_video'))  # added by Steve to feed the first frame at the first iteration
cap_truth = cv.VideoCapture(loadeddict.get('input_truth')) if loadeddict.get('input_truth') is not None else None
truth = None
while (1):
    index += 1
    # if index % 2 == 0: continue
    if 1: # index > 50:
        ok, frame = cap.read()
        _, truth = cap_truth.read() if cap_truth is not None else (None, None)
    if ok:
        smallFrame = cv.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        truthFrame = cv.cvtColor(cv.resize(truth, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR), cv.COLOR_BGR2GRAY) if truth is not None else None
        maskedFrame = np.zeros(smallFrame.shape[:2], dtype=np.uint8)
        ok, boxes = multiTracker.update(smallFrame)

        if loadeddict.get('masker') in loadeddict.get('custom_trackers'):
            maskers[0].update(frame=smallFrame)

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

            tracking_point_img = (vector[0], vector[1])
            w = vector[2]
            tracking_point_new = (int(tracking_point_img[0] / w), int(tracking_point_img[1] / w))
            # Add new position to list of points for the homographed space
            x_sequences[i].append(tracking_point_new[0])
            y_sequences[i].append(tracking_point_new[1])
            # computation of the predicted bounding box
            point1_k = (int(p1_k[0]), int(p1_k[1]))
            point2_k = (int(p2_k[0]), int(p2_k[1]))
            point1_t = (int(p1_t[0]), int(p1_t[1]))
            point2_t = (int(p2_t[0]), int(p2_t[1]))

            bbox_new = (int(point1_k[0]), int(point1_k[1]), int(point2_k[0] - point1_k[0]), int(point2_k[1] - point1_k[1]))
            bbox_new_t = (int(point1_t[0]), int(point1_t[1]), int(point2_t[0] - point1_t[0]), int(point2_t[1] - point1_t[1]))

            # RE-INITIALIZATION START
            crop_img = smallFrame[bbox_new[1]:bbox_new[1] + bbox_new[3], bbox_new[0]:bbox_new[0] + bbox_new[2]]
            hist_2, _ = np.histogram(crop_img, bins=256, range=[0, 255])
            intersection = returnIntersection(histo[i], hist_2)
            if intersection < 0:
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

            if loadeddict.get('masker') not in loadeddict.get('custom_trackers'):
                maskers[i].update(bbox=bbox_new_t, frame=smallFrame, mask=maskedFrame, color=colors[i])

            # Compute benchmark w.r.t. ground truth
            if truthFrame is not None:
                benchmarkDist.append(computeBenchmark(maskedFrame, truthFrame))

            if DEBUG:
                # cv.rectangle(smallFrame, point1_k, point2_k, colors[i], 1, 1)
                cv.rectangle(smallFrame, point1_t, point2_t, colors[i], 2, 1)

            cv.putText(smallFrame, TRACKER + ' Tracker', (100, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv.putText(smallFrame, '{:.2f}'.format(intersection), (point1_k[0], point1_k[1]-7), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv.circle(smallFrame, (int(predictedCoords[0][0]), int(predictedCoords[1][0])), 4, colors[i], -1)

            cv.circle(img, tracking_point_new, 4, colors[i], -1)
            points.write(img)  # Save video for position tracking on the basketball diagram

            # Show results
            cv.imshow('Tracking', smallFrame)
            if SHOW_MASKS:
                cv.imshow('Tracking-Masks', maskedFrame)
            if SHOW_HOMOGRAPHY:
                cv.imshow('Tracking-Homography', img)

        if 1: #index > 50:
            out.write(smallFrame)  # Save video frame by frame
            masked_image = cv.addWeighted(src1=smallFrame, alpha=0.6, src2=cv.cvtColor(maskedFrame, cv.COLOR_GRAY2RGB), beta=0.4, gamma=0)
            out_mask.write(masked_image)  # Save masked video
            out_mask.write(masked_image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv.putText(smallFrame, 'Tracking failure detected', (100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
        break

# cv.waitKey(0)
out.release()
out_mask.release()
points.release()
cv.destroyAllWindows()
end = time.time()
print(f'\nTotal time consumed for tracking: {(end - start):.2f}s')

# Show benchmark
plt.plot(benchmarkDist)
plt.xlabel("Number of Frame")
plt.ylabel("Error of Center")
plt.title("Avg. error = {}".format(int(np.mean(benchmarkDist))))
plt.tight_layout()
plt.show()

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
