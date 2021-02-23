# CSRT tracker algorithm with kalman filter
Project about object tracking for the Signal, Image and Video course

## Get started

### Importing the conda env
- Open the Anaconda shell
- Run *conda env create -f environment.yml*
- Run *conda activate name_of_the_environment*
- Open your favorite editor using the just-created environment (for example, fire *code .* in the same Anaconda shell)

### Compile C libraries
Run:
- `cd prim`
- `make`

## Original README
PROJECT ON MULTIPLE PLAYERS TRACKING \
ANDREA MONTIBELLER \
COMPUTER VISION 2018/19 \
LAST MODIFIED:12th April 2018

The scripts are written using Python 2.7.14 and OpenCV 3.4.1

In the folder Code you can find the following scripts:
- Calibration.py
    - To calibrate the camera we have used a set of chessboard images contained in the "Sources/Chessboard" folder
    - The script gives in output a file called "calibration.yaml"
- VideoCalibration.py
    - It generates an undistorted video. It accepts in input: 
        - "calibration.yaml"
        - the distorted video inside the "Sources/Video" folder
    - It gives in output an undistorted video in the "Output/Video" folder 
- Homography.py
    - It calculates the homography between the video (contained in "Output/Video") and a basketball diagram image (contained in "Sources/Map")
    - To compute the homography you have to select at least 4 corrisponding points inside the video and the basketball diagram
    - It gives in output a homography matrix: "homography.yaml"
- RealWorldPosition.py
    - This script use the homography matrix to create a correspondence between a point in the video and one in the image.
    - Then it gives you the position in meters, considering as (0,0) the point at the top left of the basketball diagram.
- Accuracy.py
    - It draws 3 horizontal lines and computes their homography. To evaluate the accuracy it calculates a Mean Square Error saved in the file: "Output/Accuracy/MSE_accuracy.txt"
 -Multitracker_players.py
    - It tracks multiple players using  the CSRT Algorithm and represents his trajectory on the basketball diagram.
    - Moreover, it gives an estimation of the trajectory length, the average speed and acceleration.
    - The outputs are saved in the Output/Tracking folder.

