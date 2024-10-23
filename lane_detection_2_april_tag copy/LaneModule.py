# with improved visual feedback, nice steering 10/23/24, 10:39 AM, final ish

import cv2
import numpy as np
import utils
 
curveList = []
avgVal=10
 
def getLaneCurve(img, display=2):
    imgCopy = img.copy()
    imgResult = img.copy()
    
    #### STEP 1: Threshold the image to detect the white lines
    imgThres = utils.thresholding(img)

    #### STEP 2: Get image dimensions and warp it for easier lane detection
    hT, wT, c = img.shape
    points = utils.valTrackbars()
    imgWarp = utils.warpImg(imgThres, points, wT, hT)
    imgWarpPoints = utils.drawPoints(imgCopy, points)

    #### STEP 3: Get the left and right lane positions
    leftBasePoint, _ = utils.getHistogram(imgWarp[:, :wT//2], display=True, minPer=0.2, region=4)
    rightBasePoint, _ = utils.getHistogram(imgWarp[:, wT//2:], display=True, minPer=0.2, region=4)
    rightBasePoint += wT // 2  # Adjust to the global coordinates of the full image
    
    # Calculate the center point between the left and right lane borders
    centerPoint = (leftBasePoint + rightBasePoint) // 2
    
    # Calculate the curve (deviation from the center of the image)
    middlePoint = wT // 2
    curveRaw = centerPoint - middlePoint
    
    dead_zone = 10  # Set the width of the dead zone in pixels

    # Implement the dead zone
    if -dead_zone < curveRaw < dead_zone:
        curveRaw = 0  # No steering adjustment if within the dead zone

    #### STEP 4: Smooth out the curve using a list to store past values
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList) / len(curveList))

    # Enhanced visual feedback
    if display != 0:
        # Create base result image with lane overlay
        imgInvWarp = utils.warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        
        # Dynamic lane coloring based on detection confidence
        lane_confidence = min(255, max(0, int(255 * (1 - abs(curve/100)))))
        imgLaneColor[:] = 0, lane_confidence, 0
        
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        
        # Drawing constants
        midY = hT - 50  # Moved lower for better visibility
        centerX = wT // 2
        
        # Draw lane boundaries
        cv2.line(imgResult, (leftBasePoint, midY), (leftBasePoint, midY + 30), (0, 255, 0), 3)
        cv2.line(imgResult, (rightBasePoint, midY), (rightBasePoint, midY + 30), (0, 255, 0), 3)
        
        # Draw center line and curve indicator
        cv2.line(imgResult, (centerX, midY), (centerX + (curve * 3), midY), (255, 0, 255), 3)
        
        # Enhanced steering visualization
        arrow_length = 80
        arrow_start = (centerX, midY - 30)
        arrow_angle = np.radians(curve * 0.5)  # Convert curve to radians for smooth rotation
        arrow_end = (
            int(arrow_start[0] + arrow_length * np.sin(arrow_angle)),
            int(arrow_start[1] - arrow_length * np.cos(arrow_angle))
        )
        
        # Dynamic arrow color based on steering intensity
        steering_intensity = min(255, abs(int(curve * 2.55)))
        if curve < 0:
            arrow_color = (0, 255 - steering_intensity, steering_intensity)  # Green to Red (Left)
        else:
            arrow_color = (steering_intensity, 255 - steering_intensity, 0)  # Red to Green (Right)
            
        # Draw steering arrow
        cv2.arrowedLine(imgResult, arrow_start, arrow_end, arrow_color, 3, tipLength=0.3)
        
        # Add telemetry
        # Curve value
        cv2.putText(imgResult, f"Curve: {curve}", (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Steering direction
        if abs(curve) < dead_zone:
            direction = "STRAIGHT"
            direction_color = (0, 255, 0)
        else:
            direction = "LEFT" if curve < 0 else "RIGHT"
            direction_color = (0, 0, 255) if abs(curve) > 50 else (0, 255, 255)
        
        cv2.putText(imgResult, f"STEERING {direction}", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, direction_color, 2)
        
        # Draw confidence meter
        confidence = 100 - min(100, abs(curve))
        cv2.rectangle(imgResult, (wT-120, 20), (wT-20, 40), (0, 0, 0), -1)
        cv2.rectangle(imgResult, (wT-120, 20), (wT-120 + int(confidence), 40), 
                     (0, 255, 0), -1)
        cv2.putText(imgResult, f"{int(confidence)}%", (wT-110, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if display == 2:
        imgStacked = utils.stackImages(0.7, ([img, imgWarpPoints, imgWarp]))
        cv2.imshow('ImageStack', imgStacked)
        cv2.imshow('Result', imgResult)
    elif display == 1:
        cv2.imshow('Result', imgResult)

    # Normalize curve value
    curve = curve / 100
    curve = max(-1, min(1, curve))  # Clamp between -1 and 1
    
    return curve


 
if __name__ == '__main__':
    cap = cv2.VideoCapture('lane_detection_2_april_tag copy/vid2.mp4')
    initialTrackBarVals = [102, 80, 20, 214]
    utils.initializeTrackbars(initialTrackBarVals)
    frameCounter = 0
    
    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
            
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.resize(img, (480, 240))
        curve = getLaneCurve(img, display=2)
        
        # Enhanced console feedback
        steering_intensity = abs(curve)
        steering_bar = "=" * int(steering_intensity * 20)
        direction = "LEFT " if curve < 0 else "RIGHT" if curve > 0 else "STRAIT"
        print(f"Steering {direction} [{steering_bar:<20}] {curve:.2f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
