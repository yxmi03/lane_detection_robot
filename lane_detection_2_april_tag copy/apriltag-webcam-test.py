import cv2
from pupil_apriltags import Detector
import numpy as np
import logging

class AprilTagDetector:
    def __init__(self):
        self.at_detector = Detector(
            families='tag36h11',
            nthreads=4,
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
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def detect_and_process_tags(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(gray)
        
        detected_tags = []
        for tag in tags:
            tag_id = tag.tag_id
            if tag_id in self.tag_actions:
                action = self.tag_actions[tag_id]
                detected_tags.append((tag_id, action, tag.center, tag.corners))
            else:
                self.logger.warning(f"Detected unknown tag ID: {tag_id}")
        
        return detected_tags

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam, change if needed
    detector = AprilTagDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            detector.logger.error("Failed to capture frame")
            break

        detected_tags = detector.detect_and_process_tags(frame)

        for tag_id, action, center, corners in detected_tags:
            # Draw tag outline
            cv2.polylines(frame, [np.int32(corners)], True, (0, 255, 0), 2)

            # Draw tag center
            cv2.circle(frame, tuple(np.int32(center)), 5, (0, 0, 255), -1)

            # Put tag ID and action text
            cv2.putText(frame, f"ID: {tag_id}", (int(center[0]) - 10, int(center[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Action: {action}", (10, 30 * (tag_id + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("AprilTag Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()