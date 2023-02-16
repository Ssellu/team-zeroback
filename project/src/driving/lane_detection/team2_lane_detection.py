#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import cv2, random, math, time
#from std_msgs.msg import Float32
from team2_pjt.msg import lane_detector_msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
Width = 640
Height = 480

bridge = CvBridge()
cv_image = np.empty(shape=[0])
cv_image_origin = np.empty(shape=[0])
roi_offset_y = 300#320#380 , 340
roi_offset_x = 5 #30
roi_gap = 40 #40
non_detection_stopline_slope = 0.2

NO_LINE_LIMIT_CNT = 3
no_line_cnt = 0
avg_lpos = []
avg_rpos = []
old_lPos = 0
old_rPos = 0

# draw lines
def draw_lines(img, lines):
    global roi_offset_y
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # print(x1, x2)
        #print (line[0])
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = cv2.line(img, (x1, y1+roi_offset_y), (x2, y2+roi_offset_y), color, 2)
    return img

# draw rectangle
def draw_rectangle(img, lpos, rpos, roi_offset_y=0):
    center = (lpos + rpos) / 2
    cv2.rectangle(img, (lpos - 5, 15 + roi_offset_y),
                       (lpos + 5, 25 + roi_offset_y),
                       (250, 255, 100), 2)
    cv2.rectangle(img, (rpos - 5, 15 + roi_offset_y),
                       (rpos + 5, 25 + roi_offset_y),
                       (100, 255, 255), 2)
    cv2.rectangle(img, (center-5, 15 + roi_offset_y),
                       (center+5, 25 + roi_offset_y),
                       (50, 150, 200), 2)    
    cv2.rectangle(img, (315, 15 + roi_offset_y),
                       (325, 25 + roi_offset_y),
                       (0, 0, 255), 2)
    return img


# left lines, right lines
def divide_left_right(lines):
    global Width

    low_slope_threshold = 0
    high_slope_threshold = 40 #20

    # calculate slope & filtering with threshold
    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            slope = 0
        else:
            slope = float(y2-y1) / float(x2-x1)
        
        if abs(slope) > low_slope_threshold and abs(slope) < high_slope_threshold:
            slopes.append(slope)
            new_lines.append(line[0])

    # divide lines left to right
    left_lines = []
    right_lines = []

    for j in range(len(slopes)):
        Line = new_lines[j]
        slope = slopes[j]

        x1, y1, x2, y2 = Line
        
        # modified slope  0 to 0.2  defency stop line
        if (slope < -non_detection_stopline_slope) and (x2 < Width/2 + 30):  #modified log 0 -> 5 -> 30  
            left_lines.append([Line.tolist()])
        elif (slope > non_detection_stopline_slope) and (x1 > Width/2 - 30): 
            right_lines.append([Line.tolist()])

    return left_lines, right_lines

# get average m, b of lines
def get_line_params(lines):
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    size = len(lines)
    if size == 0:
        return 0, 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        x_sum += x1 + x2
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)

    x_avg = float(x_sum) / float(size * 2)
    y_avg = float(y_sum) / float(size * 2)

    m = m_sum / size
    b = y_avg - m * x_avg

    return m, b

# get lpos, rpos
def get_line_pos(lines, left=False, right=False):
    m, b = get_line_params(lines)
    x1, x2 = 0, 0
    if m == 0 and b == 0:
        if left:
            pos = 0
        if right:
            pos = Width-1
    else:
        y = roi_gap / 2
        pos = (y - b) / m

        b += roi_offset_y
        x1 = (Height - b) / float(m)
        x2 = ((Height/2) - b) / float(m)
  
    return x1, x2, int(pos)

# show image and return lpos, rpos
def process_image(frame):
    global Width
    global roi_offset_y, roi_gap
    global no_line_cnt
    # gray
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # blur
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    blur_gray = cv2.GaussianBlur(blur_gray,(kernel_size, kernel_size), 0)
    blur_gray = cv2.GaussianBlur(blur_gray,(kernel_size, kernel_size), 0)
    
    #get rid of noise
    kernel = np.ones((5,5),np.uint(8))
    erode_gray = cv2.erode(blur_gray,kernel,iterations=1)
    # canny edge
    low_threshold = 60
    high_threshold = 80 # 70 
    edge_img = cv2.Canny(np.uint8(erode_gray), low_threshold, high_threshold)
    cv2.imshow("edge_img", edge_img)

    # HoughLinesP
    roi = edge_img[roi_offset_y : roi_offset_y+roi_gap, roi_offset_x : Width-roi_offset_x]
    all_lines = cv2.HoughLinesP(roi,1,math.pi/180,30,10,2)

    # divide left, right lines
    if all_lines is None:
        return (0, Width-1), frame

    left_lines, right_lines = divide_left_right(all_lines)

    # get center of lines
    lx1, lx2, lpos = get_line_pos(left_lines, left=True)
    rx1, rx2, rpos = get_line_pos(right_lines, right=True)

    if rx1 ==0 and rx2 == 0:
        rx1 = Width -1
        rx2 = Width -1
    if (rpos == 0):
        rpos = Width -1

    #To do: if no find line, then use the before point that find road line 
    #consider no find one line and two line 
    # refer to this variables( NO_LINE_LIMIT_CNT, no_line_cnt, avg_lpos, avg_rpos, old_lPos, old_rPos )
    
    frame = cv2.line(frame, (int(lx1), Height), (int(lx2), (Height/2)), (255, 0,0), 3)
    frame = cv2.line(frame, (int(rx1), Height), (int(rx2), (Height/2)), (255, 0,0), 3)

    # draw lines
    frame = draw_lines(frame, left_lines)
    frame = draw_lines(frame, right_lines)
    frame = cv2.line(frame, (230, 235), (410, 235), (255,255,255), 2)
                                 
    # draw rectangle
    frame = draw_rectangle(frame, lpos, rpos, roi_offset_y=roi_offset_y)

    # draw roi
    frame = cv2.rectangle(frame, (roi_offset_x, roi_offset_y), (Width-roi_offset_x, roi_offset_y+roi_gap), (0,0,255),2)
      
    return (lpos, rpos), frame 

def draw_steer(image, steer_angle):
    steer_angle = -steer_angle # modified  
    global Width, Height, arrow_pic

    arrow_pic = cv2.imread('/home/nvidia/xycar_ws/src/team2_pjt/src/steer_arrow.png', cv2.IMREAD_COLOR)
    origin_Height = arrow_pic.shape[0]
    origin_Width = arrow_pic.shape[1]
    steer_wheel_center = origin_Height * 0.74
    arrow_Height = Height/2
    arrow_Width = (arrow_Height * 462)/728

    #modified  new virtual_Steer_angle 
    virtual_steer_angle = steer_angle * 2.5
    if abs(virtual_steer_angle) >50:
        if virtual_steer_angle > 0:
            virtual_steer_angle = 50
        else : virtual_steer_angle = -50 # modified
    matrix = cv2.getRotationMatrix2D((origin_Width/2, steer_wheel_center), virtual_steer_angle, 0.7) #modified    
    
    arrow_pic = cv2.warpAffine(arrow_pic, matrix, (origin_Width+60, origin_Height))
    arrow_pic = cv2.resize(arrow_pic, dsize=(arrow_Width, arrow_Height), interpolation=cv2.INTER_AREA)

    gray_arrow = cv2.cvtColor(arrow_pic, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)

    arrow_roi = image[arrow_Height: Height, (Width/2 - arrow_Width/2) : (Width/2 + arrow_Width/2)]
    arrow_roi = cv2.add(arrow_pic, arrow_roi, mask=mask)
    res = cv2.add(arrow_roi, arrow_pic)
    image[(Height - arrow_Height): Height, (Width/2 - arrow_Width/2): (Width/2 + arrow_Width/2)] = res

def img_callback(data):
    global cv_image, cv_image_origin
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    cv_image_origin = bridge.imgmsg_to_cv2(data, "bgr8")
def start():
    global cv_image, Width, Height, cv_image_origin
    rospy.init_node("lane_detector", anonymous=True)
    rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback) # spin
    lane_publisher = rospy.Publisher("lane", lane_detector_msg, queue_size=10)
    rate = rospy.Rate(100)
    img_cnt = 548
    while not rospy.is_shutdown():
        if cv_image.size != (640*480*3):
           print ("didn't load")
           continue
        img_cnt+=1
        pos, frame = process_image(cv_image)
        center = (pos[0] + pos[1]) / 2 
        diff_from_center = (Width/2) - center
        steer_angle = -diff_from_center * 0.4  
        draw_steer(frame, steer_angle)
        cv2.imshow("frame",frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        msg = lane_detector_msg()
        msg.angle = -steer_angle
        msg.lpos = pos[0]
        msg.rpos = pos[1]
        msg.good_distance_to_stop = 1 # temp val. (TO DO)
        lane_publisher.publish(msg) #steer_angle
        rate.sleep()

if __name__ == '__main__':
    start()
