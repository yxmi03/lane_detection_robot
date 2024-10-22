import cv2
from pupil_apriltags import Detector
import numpy as np

class AprilTagDetector:
    def __init__(self):
        self.at_detector = Detector(
            families='tag36h11',
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        self.tag_actions = {
            0: "Railway level crossing: Stop and wait for barrier to be raised",
            1: "Watch out for pedestrians: Wait for pedestrians to pass",
            2: "Pass through intersection according to traffic lights",
            3: "Navigate around double lane obstacle",
            4: "Navigate around obstacles and continue driving"
        }

    def detect_and_process_tags(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(gray)
        
        for tag in tags:
            tag_id = tag.tag_id
            if tag_id in self.tag_actions:
                action = self.tag_actions[tag_id]
                print(f"Detected Tag ID {tag_id}: {action}")
                return action, tag_id
        return None, None

# Initialize the detector
april_tag_detector = AprilTagDetector()

def getAction(img):
    return april_tag_detector.detect_and_process_tags(img)
