import cv2 as cv
import numpy as np
import yaml
import scipy as sp
from scipy import signal
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time, sys
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
        pts.append([x, y])
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
        cv.circle(img2, (pts[-1][0], pts[-1][1]), 3, (0, 0, 255), -1)
    if len(pts) > 1:
        for i in range(len(pts) - 1):
            cv.circle(img2, (pts[i][0], pts[i][1]), 4, (0, 0, 255), -1)  # x ,y is the coordinates of the mouse click place
            cv.line(img=img2, pt1=(pts[i][0], pts[i][1]), pt2=(pts[i+1][0], pts[i+1][1]), color=(255, 0, 0), thickness=1)
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

CONFIG_FILE = 'config.yaml'
POLYNOMIAL_ROI = True
BENCHMARK_OUT = None

# Executed with custom config.yaml => bechmark computation
if len(sys.argv) > 2:
    CONFIG_FILE = sys.argv[1]
    BENCHMARK_OUT = sys.argv[2]

# Read congigurations
with open(CONFIG_FILE) as f:
    loadeddict = yaml.full_load(f)
    TRACKER = loadeddict.get('tracker')
    MASKER = loadeddict.get('masker')
    TAU = loadeddict.get('tau')
    RESIZE_FACTOR = loadeddict.get('resize_factor')
    DEBUG = loadeddict.get('debug')
    MANUAL_ROI_SELECTION = loadeddict.get('manual_roi_selection')

WINDOW_HEIGHT = 700

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
bboxes , bboxes_roni = [] , []
poly_roi = []
colors = []
histo = []

# Set output video
if DEBUG:
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    out = cv.VideoWriter(loadeddict.get('out_players'), fourcc, fps, smallFrame.shape[1::-1])
    out_mask = cv.VideoWriter(loadeddict.get('out_players_mask'), fourcc, fps/3, smallFrame.shape[1::-1])
    points = cv.VideoWriter(loadeddict.get('out_homography'), fourcc, fps, img.shape[1::-1])


#    __  __          _____ _   _
#   |  \/  |   /\   |_   _| \ | |
#   | \  / |  /  \    | | |  \| |
#   | |\/| | / /\ \   | | | . ` |
#   | |  | |/ ____ \ _| |_| |\  |
#   |_|  |_/_/    \_\_____|_| \_|


if not MANUAL_ROI_SELECTION:
    if POLYNOMIAL_ROI:
        poly_roi =  loadeddict.get('pts')
        poly_roi_frame_number =  loadeddict.get('pts_frame_numbers')
        bboxes_roni_tmp =  loadeddict.get('bboxes_roni')
        for n_target, target_selection in enumerate(poly_roi): #iterate over targets
            bboxes.append([])
            bboxes_roni.append([])
            maskers.append(getMaskerByName(loadeddict.get('masker'),
                            debug=DEBUG,
                            frame=smallFrame, 
                            config=loadeddict,
                            poly_roi=poly_roi[n_target][0] if POLYNOMIAL_ROI else None, 
                            update_mask=loadeddict.get('update_mask')
                            ))
            for n_selection , frame_selection in enumerate(target_selection): #iterate masks over time for a single target
                if not loadeddict.get('multi_selection') and n_selection > 0:
                    continue
                bbox = cv.boundingRect(np.array(frame_selection))
                bboxes[-1].append(bbox)

                crop_img = smallFrame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                hist_1, _ = np.histogram(crop_img, bins=256, range=[0, 255])
                histo.append(hist_1)
                
                colors.append(colorutils.pickNewColor(color_names_used))
                
                #read the frame where the selection was made, to train the model
                cap.set(cv.CAP_PROP_POS_FRAMES, poly_roi_frame_number[n_selection])
                ok , frame = cap.read()
                if not ok: exit("Fatal error!")
                smallFrame_succ = cv.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

                bbox_roni = maskers[n_target].addModel(frame=smallFrame_succ, 
                                                       poly_roi=poly_roi[n_target][n_selection], 
                                                       bbox=bbox, 
                                                       bbox_roni=bboxes_roni_tmp[n_target][n_selection] if bboxes_roni_tmp is not None else None, #for automatic retrivial of bbox_roni
                                                       n_frame=poly_roi_frame_number[n_selection])
                bboxes_roni[-1].append(bbox_roni)
    else:
        exit("NOT SUPPORTED RECTANGULAR SELECTION")
else:
    video_length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_to_init = int(video_length / loadeddict.get('re_init_span')) #ask for a selection every frame_to_init
    poly_roi_frame_number = []
    k = 0
    stop_selection = False
    while ok and not stop_selection:
        if (k % frame_to_init) == 0:
            print("[INFO] Selection n° {}/{}".format(int(k / frame_to_init), loadeddict.get('re_init_span')))
            poly_roi_frame_number.append(k)
            n_target = 0            
            bbox = None
            if POLYNOMIAL_ROI:
                pts = []
                cv.namedWindow('ROI')
                cv.imshow('ROI', smallFrame)
                cv.setMouseCallback('ROI', drawPolyROI, {"image": smallFrame, "alpha": 0.6})
                print("[INFO] Click the left button: select the point, right click: delete the last selected point, click the middle button: inspect the ROI area")
                print("[INFO] Press ENTER to determine the selection area and save it")
                print("[INFO] Press q or ESC to quit")
            stop = False
            while not stop:
                if POLYNOMIAL_ROI:
                    pass
                else:
                    bbox = cv.selectROI('ROI', smallFrame, False)
                    if bbox == (0, 0, 0, 0):  # no box selected
                        cv.destroyWindow('ROI')
                        bbox = None
                        stop = True
                    print('[INFO] Press q to quit selecting boxes and start tracking, or any other key to select next object')

                if bbox:  # because the callback of the mouse does not block the main thread
                    crop_img = smallFrame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                    if k < frame_to_init: #first iteration, bboxes is still equal to [] 
                        bboxes.append([])
                        bboxes_roni.append([])
                        #now init a masker for every target in the first selection. For the subsequent selections, I'll call a method for fitting the multiple models
                        maskers.append(getMaskerByName(loadeddict.get('masker'),
                                        debug=DEBUG,
                                        frame=smallFrame,
                                        config=loadeddict,
                                        poly_roi=poly_roi[n_target][0] if POLYNOMIAL_ROI else None, 
                                        update_mask=loadeddict.get('update_mask')
                                        ))
                    if n_target >= len(bboxes):
                        print("[ERROR] The number of selection can not be great to the number of selections at the first frame")
                    else:
                        bboxes[n_target].append(bbox)
                        colors.append(colorutils.pickNewColor(color_names_used))
                        hist_1, _ = np.histogram(crop_img, bins=256, range=[0, 255])
                        histo.append(hist_1)
                        bbox_roni = maskers[n_target].addModel(frame=smallFrame, poly_roi=poly_roi[n_target][-1], bbox=bbox, n_frame=k)
                        bboxes_roni[n_target].append(bbox_roni)
                        bbox = None
                        n_target += 1
                else:
                    time.sleep(0.2)

                key = cv.waitKey(0) & 0xFF
                if (key == ord('q')):  # q is pressed
                    cv.destroyWindow('ROI')
                    stop = True
                    if not loadeddict.get('multi_selection'):
                        stop_selection = True
                if POLYNOMIAL_ROI and key == ord("\r"):
                    if len(pts) >= 3:
                        if k < frame_to_init:
                            poly_roi.append([])
                        if n_target >= len(poly_roi):
                            print("[ERROR] The number of selection can not be great to the number of selections at the first frame")
                        else:
                            poly_roi[n_target].append(pts)
                        bbox = cv.boundingRect(np.array(pts))  # extract the minimal Rectangular that fit the polygon just selected. This because Tracking algos work with rect. bbox
                        pts = []
                    else:
                        print("Not enough points selected")
        k += 1
        ok, frame = cap.read()
        if ok:
            smallFrame = cv.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    
print('Selected poly roi: {}\n'.format(poly_roi))
print('Selected bboxes_roi: {}\n'.format(bboxes_roni))
if poly_roi_frame_number is not None:
    print('Selected frames: {}\n'.format(poly_roi_frame_number))

#re-init video reader
cap.set(cv.CAP_PROP_POS_FRAMES, 0)
ok , frame = cap.read()
if not ok: exit("Fatal error!")
smallFrame = cv.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

multiTracker = cv.legacy.MultiTracker_create()

# List for saving points of tracking in the basketball diagram (homography)
x_sequence_image, y_sequence_image = [], []
x_sequences, y_sequences = [], []
for n_target in range(len(bboxes)):
    bbox = bboxes[n_target][0]
    multiTracker.add(createTracker(TRACKER), smallFrame, bbox)
    x_sequences.append([])
    y_sequences.append([])

    kalman_filters.append(KalmanFilter())
    kalman_filtersp1.append(KalmanFilter())
    kalman_filtersp2.append(KalmanFilter())

    tracking_point = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3]))
    cv.circle(smallFrame, tracking_point, 4, (255, 200, 0), -1)
    cv.rectangle(smallFrame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
    # Compute the point in the homographed space: destination point(image)=homography matrix*source point(video)
    vector = np.dot(h, np.transpose([tracking_point[0], tracking_point[1], 1]))
    # Evaluation of the vector
    tracking_point_img = (vector[0], vector[1])
    w = vector[2]
    tracking_point_new = (int(tracking_point_img[0] / w), int(tracking_point_img[1] / w))
    x_sequences[n_target].append(tracking_point_new[0])
    y_sequences[n_target].append(tracking_point_new[1])
    cv.circle(img, tracking_point_new, 4, colors[n_target], -1)

if DEBUG:
    # Save and visualize the chosen bounding box and its point used for homography
    cv.imwrite(loadeddict.get('out_bboxes'), smallFrame)
    cv.putText(smallFrame, 'Selected Bounding Boxes. PRESS SPACE TO CONTINUE...', (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv.namedWindow('Tracking')
    #cv.resizeWindow('Tracking', WINDOW_WIDTH,  WINDOW_HEIGHT)
    cv.imshow('Tracking', smallFrame)
    cv.waitKey(0)

if DEBUG and loadeddict.get('show_masks'):
    cv.namedWindow('Tracking-Masks', cv.WINDOW_NORMAL)
    cv.resizeWindow('Tracking-Masks', WINDOW_WIDTH,  WINDOW_HEIGHT)
if DEBUG and loadeddict.get('show_homography'):
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
        maskedFrame = np.zeros_like(smallFrame, dtype=np.uint8) #to have the output mask red, compile only the last channel of the last dimension
        ok, boxes = multiTracker.update(smallFrame)

        if loadeddict.get('masker') in loadeddict.get('custom_trackers'):
            maskers[0].update(frame=smallFrame, mask=maskedFrame)

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
            with np.errstate(divide='ignore', invalid='ignore'):
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
                status = maskers[i].update(bbox=bbox_new_t, frame=smallFrame, mask=maskedFrame, color=colors[i])
                if status is not None: #re-init the Tracker
                    print('RE-INITIALIZE TRACKER n° %d' % status) 
                    multiTracker = cv.legacy.MultiTracker_create()        
                    if loadeddict.get('masker') == 'GrabCut':
                        if loadeddict.get('show_masks'):
                            cv.imshow('Tracking-Masks', maskedFrame[:,:,2])
                        maskers= []
                        for n_target in range(len(bboxes)):
                            # Request new mask selection
                            pts = []
                            cv.namedWindow('ROI')
                            cv.imshow('ROI', smallFrame)
                            cv.setMouseCallback('ROI', drawPolyROI, {"image": smallFrame, "alpha": 0.6})
                            print("[INFO] Click the left button: select the point, right click: delete the last selected point, click the middle button: inspect the ROI area")
                            print("[INFO] Press ENTER to determine the selection area and save it")
                            print("[INFO] Press q or ESC to quit")
                            while True:
                                key = cv.waitKey(0) & 0xFF
                                if key == ord(' ') or key == ord("\r"):  # q or enter is pressed
                                    cv.destroyWindow('ROI')
                                    break
                            maskers.append(getMaskerByName(loadeddict.get('masker'),
                                debug=DEBUG,
                                frame=smallFrame, 
                                config=loadeddict,
                                poly_roi=pts
                            ))
                            bbox = cv.boundingRect(np.array(pts))
                            maskedFrame = np.zeros_like(smallFrame, dtype=np.uint8)
                            maskers[n_target].update(bbox=bbox, frame=smallFrame, mask=maskedFrame, color=colors[n_target])
                            multiTracker.add(createTracker(TRACKER), smallFrame, bbox)
                            pts = []
                    else: 
                        for n_target in range(len(bboxes)):
                            nb = bboxes[n_target][status]
                            multiTracker.add(createTracker(TRACKER), smallFrame, nb)

            # Compute benchmark w.r.t. ground truth
            if truthFrame is not None:
                benchmarkDist.append(computeBenchmark(maskedFrame[:,:,2], truthFrame))

            if DEBUG:
                # cv.rectangle(smallFrame, point1_k, point2_k, colors[i], 1, 1)
                cv.rectangle(smallFrame, point1_t, point2_t, colors[i], 2, 1)
                cv.putText(smallFrame, TRACKER + ' Tracker', (100, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
                cv.putText(smallFrame, '{:.2f}'.format(intersection), (point1_k[0], point1_k[1]-7), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                points.write(img)  # Save video for position tracking on the basketball diagram

                # Show results
                cv.imshow('Tracking', smallFrame)
                if loadeddict.get('show_masks'):
                    cv.imshow('Tracking-Masks', maskedFrame[:,:,2])
                if loadeddict.get('show_homography'):
                    cv.imshow('Tracking-Homography', img)

        if DEBUG:
            out.write(smallFrame)  # Save video frame by frame
            out_mask.write(cv.addWeighted(src1=smallFrame, alpha=0.6, src2=maskedFrame, beta=0.4, gamma=0))  # Save masked video

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        cv.putText(smallFrame, 'Tracking failure detected', (100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
        break

end = time.time()
tot_time = end - start

if BENCHMARK_OUT is not None:
    with open(BENCHMARK_OUT, 'w') as f:
        f.write(f'{np.mean(benchmarkDist)};{tot_time}')

if DEBUG:
    out.release()
    out_mask.release()
    points.release()
cv.destroyAllWindows()

if DEBUG:
    print(f'\nTotal benchmark score: {np.mean(benchmarkDist)}')
    print(f'Total time consumed for tracking: {(end - start):.2f}s')
    
    # Show outlier scores
    if hasattr(maskers[0], 'scores'):
        plt.plot(maskers[0].scores)
        plt.xlabel("Number of Frame")
        plt.ylabel("Score")
        plt.title("Outlier score distribution")
        plt.tight_layout()
        plt.show()

    # Show benchmark
    plt.plot(benchmarkDist)
    plt.xlabel("Number of Frame")
    plt.ylabel("Error of Center")
    plt.title("Avg. error = {}".format(int(np.mean(benchmarkDist))))
    plt.tight_layout()
    plt.show()

#plt.hist([m.distances for m in maskers], bins=np.unique([m.distances for m in maskers]).size)
# plt.show()

if DEBUG:
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
    if loadeddict.get('show_homography'):
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
