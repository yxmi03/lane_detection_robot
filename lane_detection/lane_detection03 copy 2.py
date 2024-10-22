import cv2
import numpy as np
import math

class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


def convert_to_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def detect_edges(frame):
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imshow("Edges", edges)
    return edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Define the region of interest (lower half)
    polygon = np.array([[ 
        (0, height),
        (0, height/2),
        (width, height/2),
        (width, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    cv2.imshow("ROI", cropped_edges)
    return cropped_edges

def detect_line_segments(cropped_edges):
    rho = 1  
    theta = np.pi / 180  
    min_threshold = 10
    line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold,
                                    np.array([]), minLineLength=5, maxLineGap=10)
    return line_segments

def average_slope_intercept(frame, line_segments):
    lane_lines = []

    if line_segments is None:
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []
    boundary = 1 / 3

    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue  # Skip vertical lines

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]

            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    if left_fit:
        lane_lines.append(make_points(frame, np.average(left_fit, axis=0)))
    if right_fit:
        lane_lines.append(make_points(frame, np.average(right_fit, axis=0)))

    return lane_lines

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 / 2)

    if slope == 0:
        slope = 0.1    

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6):
    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Ensure x1, y1, x2, y2 are integers before drawing
                if all(isinstance(coord, int) for coord in (x1, y1, x2, y2)):
                    cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)

    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

def find_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(frame, contours):
    for contour in contours:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
    return frame

def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    return cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)
    
def apply_dead_zone(value, dead_zone):
    if abs(value) < dead_zone:
        return 0
    return value

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None

    def update(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * value + (1 - self.alpha) * self.value
        return self.value
    
def calculate_steering_angle(frame, lane_lines):
    if len(lane_lines) == 0:
        return 0

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = 0.02 # 2%
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    # Calculate angle
    y_offset = int(height / 2)
    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  
    steering_angle = angle_to_mid_deg + 90

    return steering_angle

# Initialize the low-pass filter
steering_filter = LowPassFilter(alpha=0.2)  # Adjust alpha as needed (0.1 to 0.3 is a good range)

# Initialize the moving average
steering_average = MovingAverage(5)  # Use a window of 5 frames

# PID constants
Kp = 0.03
Ki = 0.001
Kd = 0.02
pid_controller = PID(Kp, Ki, Kd)

# Video capture settings
video = cv2.VideoCapture('C:/Users/yeiru/Documents/autonomous_robot/lane_detection_2_april_tag copy/vid2.mp4')
video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

prev_time = cv2.getTickCount()
steering_angle = 90  # Initialize to center

# Initialize the low-pass filter
steering_filter = LowPassFilter(alpha=0.2)

def apply_dead_zone(value, dead_zone):
    if abs(value) < dead_zone:
        return 0
    return value

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame")
        break

    gray = convert_to_gray(frame)
    edges = detect_edges(gray)
    roi = region_of_interest(edges)
    line_segments = detect_line_segments(roi)
    lane_lines = average_slope_intercept(frame, line_segments)

    if lane_lines:
        new_steering_angle = calculate_steering_angle(frame, lane_lines)
        current_time = cv2.getTickCount()
        dt = (current_time - prev_time) / cv2.getTickFrequency()
        
        error = new_steering_angle - 90  # 90 is center
        error = apply_dead_zone(error, 20)  # Increased dead zone
        
        if error != 0:  # Only update if outside dead zone
            pid_output = pid_controller.update(error, dt)
            steering_angle = 90 + pid_output  # Center + adjustment
        
        steering_angle = steering_filter.update(steering_angle)
        prev_time = current_time

        # Determine direction based on steering angle
        if abs(steering_angle - 90) < 10:  # Close to center
            print("Go Straight")
        elif steering_angle > 90:
            print("Turn Right")
        else:
            print("Turn Left")

    lane_lines_image = display_lines(frame, lane_lines) if lane_lines else frame
    heading_image = display_heading_line(lane_lines_image, steering_angle)

    cv2.imshow("Lane Tracking", heading_image)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

video.release()
cv2.destroyAllWindows()