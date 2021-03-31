import cv2 as cv
import numpy as np
import yaml
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
import time, sys, os
import colorutils
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
    RESIZE_FACTOR = loadeddict.get('resize_factor')
    DEBUG = loadeddict.get('debug')
    MANUAL_ROI_SELECTION = loadeddict.get('manual_roi_selection')

WINDOW_HEIGHT = 700

# Set input video
cap = cv.VideoCapture(loadeddict.get('input_video'))
ratio = cap.get(cv.CAP_PROP_FRAME_WIDTH) / cap.get(cv.CAP_PROP_FRAME_HEIGHT)
WINDOW_WIDTH = int(WINDOW_HEIGHT * ratio)
if not cap.isOpened():
    exit("Input video not opened correctly")
ok, frame = cap.read()
smallFrame = cv.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
maskers = []
color_names_used = set()
bboxes , bboxes_roni = [] , []
poly_roi = []
colors = []

# Create output folder if mising
for out_file in [loadeddict.get('out_tracked'), loadeddict.get('out_mask'), loadeddict.get('out_binary_mask')]:
    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

# Set output video
if DEBUG:
    fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    out = cv.VideoWriter(loadeddict.get('out_tracked'), fourcc, fps, smallFrame.shape[1::-1])
    out_mask = cv.VideoWriter(loadeddict.get('out_mask'), fourcc, fps/3, smallFrame.shape[1::-1])
    out_mask_binary = cv.VideoWriter(loadeddict.get('out_binary_mask'), fourcc, fps/3, smallFrame.shape[1::-1])


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
for n_target in range(len(bboxes)):
    bbox = bboxes[n_target][0] #init the tracker with the first selection of each target
    multiTracker.add(createTracker(TRACKER), smallFrame, bbox)

if DEBUG:
    # Save and visualize the chosen bounding box and its point used for homography
    cv.putText(smallFrame, 'PRESS SPACE TO START', (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv.namedWindow('Tracking')
    #cv.resizeWindow('Tracking', WINDOW_WIDTH,  WINDOW_HEIGHT)
    cv.imshow('Tracking', smallFrame)
    cv.waitKey(0)

if DEBUG and loadeddict.get('show_masks'):
    cv.namedWindow('Tracking-Masks', cv.WINDOW_NORMAL)
    cv.resizeWindow('Tracking-Masks', WINDOW_WIDTH,  WINDOW_HEIGHT)

benchmarkDist = []
start = time.time()
index = 0
cap = cv.VideoCapture(loadeddict.get('input_video'))  # added by Steve to feed the first frame at the first iteration
cap_truth = cv.VideoCapture(loadeddict.get('input_truth')) if loadeddict.get('input_truth') is not None else None
truth = None
while 1:
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
            point1_t = (int(p1_t[0]), int(p1_t[1]))
            point2_t = (int(p2_t[0]), int(p2_t[1]))

            bbox_new_t = (int(point1_t[0]), int(point1_t[1]), int(point2_t[0] - point1_t[0]), int(point2_t[1] - point1_t[1]))

            if loadeddict.get('masker') not in loadeddict.get('custom_trackers'):
                originalFrame = smallFrame.copy()
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
                            cv.imshow('ROI', originalFrame)
                            cv.setMouseCallback('ROI', drawPolyROI, {"image": originalFrame, "alpha": 0.6})
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
                            maskedFrame = np.zeros_like(originalFrame, dtype=np.uint8)
                            maskers[n_target].update(bbox=bbox, frame=originalFrame, mask=maskedFrame, color=colors[n_target])
                            multiTracker.add(createTracker(TRACKER), originalFrame, bbox)
                            smallFrame = originalFrame
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

                # Show results
                cv.imshow('Tracking', smallFrame)
                if loadeddict.get('show_masks'):
                    cv.imshow('Tracking-Masks', maskedFrame[:,:,2])

        if DEBUG:
            out.write(smallFrame)  # Save video frame by frame
            out_mask.write(cv.addWeighted(src1=smallFrame, alpha=0.6, src2=maskedFrame, beta=0.4, gamma=0))  # Save masked video
            out_mask_binary.write(maskedFrame)

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
    out_mask_binary.release()
cv.destroyAllWindows()

if DEBUG:
    print(f'\nTotal benchmark score: {np.mean(benchmarkDist)}')
    print(f'Total time consumed for tracking: {(end - start):.2f}s')
    
    # Show benchmark
    plt.plot(benchmarkDist)
    plt.xlabel("Number of Frame")
    plt.ylabel("IoU")
    plt.title("Avg. error = {}".format(int(np.mean(benchmarkDist))))
    plt.tight_layout()
    plt.show()