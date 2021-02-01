"This script create some images useful for the presentation"

import cv2

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('../Output/Video/undistorted.primo_tempo.asf')

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Corner detection in the image
img = cv2.imread('../Sources/Map/basket_field.jpg')
index=1
font = cv2.FONT_HERSHEY_SIMPLEX


def draw_circle_video(event,x,y,flags,param):
    global index
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(smallFrame,(x,y),4,(255,0,0),-1)
        cv2.putText(smallFrame, str(index), (x+4, y+4), font, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
        index=index+1

def draw_circle_image(event,x,y,flags,param):
    global index
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),4,(0,80,255),-1)
        cv2.putText(img, str(index), (x+4, y+4), font, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        index=index+1

# Save image
if cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        smallFrame = cv2.resize(frame, (0, 0), fy=0.35, fx=0.35)
        cv2.imshow('Frame', smallFrame)
        #cv2.imwrite('../Output/Calibration/video_undistorted.png',frame)

        while (1):
            cv2.imshow('frame', smallFrame)
            # To end the point acquisition click 'q'
            if cv2.waitKey(20) & 0xFF == ord('q'):
                # Press Q on keyboard to  exit
                break

if cap.isOpened():
    ret, frame = cap.read()
    smallFrame = cv2.resize(frame, (0, 0), fy=0.35, fx=0.35)
    height, width, channels = smallFrame.shape
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', draw_circle_video)
    while (1):
        cv2.imshow('frame', smallFrame)
        # To end the point acquisition click 'q'
        if cv2.waitKey(20) & 0xFF == ord('q'):
            # Save the image for homography
            cv2.imwrite('../Output/Homography/Matrix.Computation/Source_points11.png',smallFrame)
            break

index=1
while (1):
    cv2.setMouseCallback('image', draw_circle_image, img)
    cv2.imshow('image', img)
    # To end the point acquisition click 'q'
    if cv2.waitKey(20) & 0xFF == ord('q'):
        # Save the image for homography
        cv2.imwrite('../Output/Homography/Matrix.Computation/Destination_points11.png', img)
        break
# When everything done, release the video capture object


i=0
# Read until video is completed
cv2.destroyAllWindows()


#while cap.isOpened():
    # Capture frame-by-frame
 #   ret, frame = cap.read()
  #  if ret == True:
   #     i=i+1
        # Display the resulting frame
    #    smallFrame = cv2.resize(frame, (0, 0), fy=0.35, fx=0.35)
     #   cv2.imshow('Frame', smallFrame)
      #  if i == 2200:
       #     cv2.imwrite('image2.png',frame)
        # Press Q on keyboard to  exit
        #if cv2.waitKey(5) & 0xFF == ord('q'):
         #   break

    # Break the loop
    #else:
     #   break



#cap.release()

# Closes all the frames
#cv2.destroyAllWindows()