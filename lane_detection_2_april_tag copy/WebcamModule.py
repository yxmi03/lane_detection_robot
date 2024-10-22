import cv2

lane_cam = cv2.VideoCapture(0)  # Camera for lane following
tag_cam = cv2.VideoCapture(1)   # Camera for AprilTag detection

def getImg(display=False, size=[480, 240]):
    _, img_lane = lane_cam.read()
    _, img_tag = tag_cam.read()
    
    img_lane = cv2.resize(img_lane, (size[0], size[1]))
    img_tag = cv2.resize(img_tag, (size[0], size[1]))
    
    if display:
        cv2.imshow('Lane Camera', img_lane)
        cv2.imshow('Tag Camera', img_tag)
    
    return img_lane, img_tag

if __name__ == '__main__':
    while True:
        img_lane, img_tag = getImg(True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

lane_cam.release()
tag_cam.release()
cv2.destroyAllWindows()