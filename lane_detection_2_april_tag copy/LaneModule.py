import cv2
import numpy as np
import utils
 
curveList = []
avgVal=10
 
def getLaneCurve(img, display=2):
    imgCopy = img.copy()
    imgResult = img.copy()
    #### STEP 1
    imgThres = utils.thresholding(img)

    #### STEP 2
    hT, wT, c = img.shape
    points = utils.valTrackbars()
    imgWarp = utils.warpImg(imgThres, points, wT, hT)
    imgWarpPoints = utils.drawPoints(imgCopy, points)

    #### STEP 3
    middlePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.5, region=4)
    curveAveragePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.9)
    
    # Calculate the curve (deviation from center)
    curveRaw = curveAveragePoint - middlePoint

    #### SETP 4
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList) / len(curveList))

    #### STEP 5: Determine direction based on curve value
    if display != 0:
        imgInvWarp = utils.warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        midY = 450
        cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
        cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
        
        # Add lines for visual feedback
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                     (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)

    if display == 2:
        imgStacked = utils.stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                             [imgHist, imgLaneColor, imgResult]))
        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        cv2.imshow('Result', imgResult)

    #### NORMALIZATION
    curve = curve / 100
    if curve > 1: curve = 1
    if curve < -1: curve = -1

    return curve

 
 
if __name__ == '__main__':
    cap = cv2.VideoCapture('C:/Users/yeiru/Documents/autonomous_robot/lane_detection_2_april_tag copy/vid2.mp4')

    intialTrackBarVals = [102, 80, 20, 214]
    utils.initializeTrackbars(intialTrackBarVals)
    frameCounter = 0

    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        success, img = cap.read()
        img = cv2.resize(img, (480, 240))
        curve = getLaneCurve(img, display=2)
        print(curve)

        # Control the motors based on the curve value
        if curve > 0.1:  # Turn right
            # Code to turn right (e.g., reduce left wheel speed)
            print("Turning Right")
        elif curve < -0.1:  # Turn left
            # Code to turn left (e.g., reduce right wheel speed)
            print("Turning Left")
        else:  # Move forward
            # Code to move forward
            print("Moving Forward")

        cv2.imshow('Vid', img)
        cv2.waitKey(1)
