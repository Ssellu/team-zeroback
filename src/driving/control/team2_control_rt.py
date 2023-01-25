#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
from std_msgs.msg import Float32
from xycar_msgs.msg import xycar_motor
motor_control = xycar_motor()
rospy.init_node("Control",anonymous=False)
pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=10)


def msg_callback(msg):
    print("recieved", msg)
    print ("receive: ", msg.data)
    global pub
    global motor_control
	
    motor_control.angle = msg.data
    motor_control.speed = 3
    pub.publish(motor_control)
print("asdasd")    
sub = rospy.Subscriber("lane", Float32, msg_callback, queue_size = 10)

rospy.spin()
