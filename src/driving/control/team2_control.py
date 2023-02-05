#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np

from team2_pjt.msg import lane_detector_msg
from xycar_msgs.msg import xycar_motor
from PID import *
motor_control = xycar_motor()
rospy.init_node("Control",anonymous=False)
pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=10)

def msg_callback(lane_msg):
    print("control node receivied  msg from lane_node")   
    print ("lane_lpos : ", lane_msg.lpos)
    print ("lane_rpos : ", lane_msg.rpos )
    print ("angle: ", lane_msg.angle)
    print ("is stop (temp, Todo): ", lane_msg.good_distance_to_stop)
    pub.publish(motor_control)


##pid = PID(0.5,0.0005,0.1)
# def msg_callback(angle_msg):
#     #print("recieved", angle_msg)
#     #print ("received: ", angle_msg.data)
#     global pub
#     global motor_control
    
#     # pid_data =  pid.pid_control(angle_msg.data)
#     # if pid_data > 50:
#     #     pid_data = 50
#     # if pid_data < -50: 
#     #     pid_data = -50
#     #print ("pid data: ", pid_data)
#     motor_control.angle = angle_msg.data * 0.4#angle_msg.data  ,  pid_data
#     print ("angle:", motor_control.angle)
#     motor_control.speed = 6 # 3   todo: 4 ~ 8
#     pub.publish(motor_control)
 
sub = rospy.Subscriber("lane", lane_detector_msg, msg_callback, queue_size = 10)
rospy.spin()
