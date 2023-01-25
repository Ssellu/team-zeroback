#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2, random, math, time
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
Width = 640
Height = 480
Offset = 320#380 , 340
Gap = 40
bridge = CvBridge()
cv_image = np.empty(shape=[0])
cv_image_origin = np.empty(shape=[0])
lst = []
CENTER_THRESHOLD = 80

# draw lines
def draw_lines(img, lines):
    global Offset
    for line in lines:
        x1, y1, x2, y2 = line[0]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = cv2.line(img, (x1, y1+Offset), (x2, y2+Offset), color, 2)
    return img

# draw rectangle
def draw_rectangle(img, lpos, rpos, offset=0):
    center = (lpos + rpos) / 2
    cv2.rectangle(img, (lpos - 5, 15 + offset),
                       (lpos + 5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (rpos - 5, 15 + offset),
                       (rpos + 5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (center-5, 15 + offset),
                       (center+5, 25 + offset),
                       (0, 255, 0), 2)    
    cv2.rectangle(img, (315, 15 + offset),
                       (325, 25 + offset),
                       (0, 0, 255), 2)
    return img


# left lines, right lines
def divide_left_right(lines):
    global Width

    low_slope_threshold = 0
    high_slope_threshold = 10

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

        if (slope < 0) and (x2 < Width/2 - 90):
            left_lines.append([Line.tolist()])       
        elif (slope > 0) and (x1 > Width/2 + 90):
            right_lines.append([Line.tolist()])

    return left_lines, right_lines

# get average m, b of lines
def get_line_params(lines):
    # sum of x, y, m
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
    global Width, Height
    global Offset, Gap

    m, b = get_line_params(lines)
    print ("ddd1")
    x1, x2 = 0, 0
    if m == 0 and b == 0:
        if left:
            pos = 0
        if right:
            pos = Width
    else:
        y = Gap / 2
        pos = (y - b) / m

        b += Offset
        x1 = (Height - b) / float(m)
        x2 = ((Height/2) - b) / float(m)
  
    return x1, x2, int(pos)

# show image and return lpos, rpos
def process_image(frame):
    global Width
    global Offset, Gap
    # gray
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # blur
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    blur_gray = cv2.GaussianBlur(blur_gray,(kernel_size, kernel_size), 0)
    blur_gray = cv2.GaussianBlur(blur_gray,(kernel_size, kernel_size), 0)

    # canny edge
    low_threshold = 60
    high_threshold = 70
    edge_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)
    cv2.imshow("edge_img", edge_img)

    # HoughLinesP
    roi = edge_img[Offset : Offset+Gap, 0 : Width]
    cv2.imshow("roi", roi)
    all_lines = cv2.HoughLinesP(roi,1,math.pi/180,30,30,10)
    

    # divide left, right lines
    if all_lines is None:
        return (0, 640), frame

    left_lines, right_lines = divide_left_right(all_lines)

    # get center of lines
    lx1, lx2, lpos = get_line_pos(left_lines, left=True)
    rx1, rx2, rpos = get_line_pos(right_lines, right=True)

    frame = cv2.line(frame, (int(lx1), Height), (int(lx2), (Height/2)), (255, 0,0), 3)
    frame = cv2.line(frame, (int(rx1), Height), (int(rx2), (Height/2)), (255, 0,0), 3)

    # draw lines
    frame = draw_lines(frame, left_lines)
    frame = draw_lines(frame, right_lines)
    frame = cv2.line(frame, (230, 235), (410, 235), (255,255,255), 2)
                                 
    # draw rectangle
    frame = draw_rectangle(frame, lpos, rpos, offset=Offset)
    cv2.imshow("ROI_FRAME",frame)

    # draw roi
    frame = cv2.rectangle(frame, (0, Offset), (Width, Offset+Gap), (0,0,255),2)
      
    return (lpos, rpos), frame 

def draw_steer(image, steer_angle):
    global Width, Height, arrow_pic

    arrow_pic = cv2.imread('/home/nvidia/xycar_ws/src/team2_pjt/src/steer_arrow.png', cv2.IMREAD_COLOR)
   
    origin_Height = arrow_pic.shape[0]
    origin_Width = arrow_pic.shape[1]
    steer_wheel_center = origin_Height * 0.74
    arrow_Height = Height/2
    arrow_Width = (arrow_Height * 462)/728
    #print ("arrow_Height : ", arrow_Height)
    #print ("arrow_Width : ", arrow_Width)
    matrix = cv2.getRotationMatrix2D((origin_Width/2, steer_wheel_center), (steer_angle) * 2.5, 0.7)    
    arrow_pic = cv2.warpAffine(arrow_pic, matrix, (origin_Width+60, origin_Height))
    arrow_pic = cv2.resize(arrow_pic, dsize=(arrow_Width, arrow_Height), interpolation=cv2.INTER_AREA)

    gray_arrow = cv2.cvtColor(arrow_pic, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)

    arrow_roi = image[arrow_Height: Height, (Width/2 - arrow_Width/2) : (Width/2 + arrow_Width/2)]
    arrow_roi = cv2.add(arrow_pic, arrow_roi, mask=mask)
    res = cv2.add(arrow_roi, arrow_pic)
    image[(Height - arrow_Height): Height, (Width/2 - arrow_Width/2): (Width/2 + arrow_Width/2)] = res

    cv2.imshow('steer', image)


def img_callback(data):
    global cv_image, cv_image_origin
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    cv_image_origin = bridge.imgmsg_to_cv2(data, "bgr8")
def start():
    global cv_image, Width, Height, cv_image_origin, lst
    rospy.init_node("lane_detector", anonymous=True)
    rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback) # spin
    
    pub = rospy.Publisher("lane", Float32, queue_size=10)
    rate = rospy.Rate(100)
   

    #cap = cv2.VideoCapture(0) # 'hough_track.avi' track2.avi
    img_cnt = 0
    
    print "789"
    failed_cnt = 0

    while not rospy.is_shutdown():
        #if not cap.isOpened(): continue  
        #file_path = "/home/nvidia/xycar_ws/data_imgs/" + str(img_cnt)+ ".JPEG"
        print "123"
        if cv_image.size != (640*480*3):
            print "didn't load"
            continue
     
        #cv2.imwrite(file_path, cv_image_origin)
        #cv2.waitKey(1)
        #fram2 = frame
    
       
        img_cnt += 1
        cv2.imshow("cv_img",cv_image)
        #cv2.waitKey(1)

        #ret, image = cap.read()

        #if ret == 1:
        #    print "success"
        #else: 
        #    print"failed"
        #    failed_cnt+=1
        #if failed_cnt == 100: return


        # if ret == 0:
        #    continue
        
        #time.sleep(500)
        
        pos, frame = process_image(cv_image)
        
        
        center = (pos[0] + pos[1]) / 2 
        center_avg = np.mean(lst)
        if abs(center - center_avg) > CENTER_THRESHOLD:
            continue
        if len(lst) > 0:
            lst.pop(0)
        lst.append(center)
        angle = (Width/2) - center
        #real angle range : -20~20
        #xycar angle setting capa: -50 ~50
        steer_angle = -angle * 0.4
        print("angle :", angle)
      
        draw_steer(frame, steer_angle)
        cv2.imshow("frame",frame)
        # steer_angle 또는 angle  제어에 보내기
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            #out.release() # s
            break
        pub.publish(steer_angle)
        print("11111111111")
        rate.sleep()
        #rospy.spin()  # spin 하면 멈춤. 
        
        

if __name__ == '__main__':
    print("00000")
    start()
    print "123"
