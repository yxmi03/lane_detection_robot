from MotorModule import Motor
from LaneModule import getLaneCurve
import WebcamModule
import cv2
import AprilTagModule

##################################################
motor = Motor(2,3,4,17,22,27)
##################################################

def main():
    img = WebcamModule.getImg()
    curveVal = getLaneCurve(img, 1)

    # Get AprilTag action
    action, tag_id = AprilTagModule.getAction(img)

    if action:
        # Handle the detected AprilTag action
        if tag_id == 1:
            # Stop at railway crossing
            motor.stop()
            # You might want to add additional logic here to wait for the barrier
        elif tag_id == 2:
            # Watch for pedestrians
            motor.move(0.1, 0, 0.1)  # Slow down
            # Add logic to detect pedestrians and wait if necessary
        elif tag_id == 3:
            # Follow traffic light instructions
            # Add logic to detect and respond to traffic lights
            pass
        elif tag_id == 4:
            # Navigate around obstacle
            # Add obstacle avoidance logic
            pass
    else:
        # Normal lane following behavior
        sen = 1.3  # SENSITIVITY
        maxVAl = 0.3 # MAX SPEED
        if curveVal > maxVAl: curveVal = maxVAl
        if curveVal < -maxVAl: curveVal = -maxVAl
        
        if curveVal > 0:
            sen = 1.7
            if curveVal < 0.05: curveVal = 0
        else:
            if curveVal > -0.08: curveVal = 0
        motor.move(0.20, -curveVal * sen, 0.05)
    
    cv2.waitKey(1)

if __name__ == '__main__':
    while True:
        main()