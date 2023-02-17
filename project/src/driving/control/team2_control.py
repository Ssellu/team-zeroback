#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import time
import rospy
from std_msgs.msg import Float32
from object_detection.traffic_sign import *
from xycar_msgs.msg import xycar_motor
from msg import BoundingBox, BoundingBoxes
"""
    Subscribe to
    - Object Detection Node
    - Lane/StopLine Detection Node

    Publish to
    - xycar_motor
"""


class MainControl:
    NO_OBJECT = -1
    LEFT_SIGN = 0
    RIGHT_SIGN = 1
    STOP_SIGN = 2
    CROSSWALK_SIGN = 3
    UTURN_SIGN = 4
    TRAFFIC_LIGHT_SIGN = 5

    SPEED = 3
    DURATION = 30 * SPEED
    ELAPSED_TIME = -1

    def __init__(self):

        rospy.init_node("Control", anonymous=False)

        self.object_detector = rospy.Subscriber(
            "object_detector", BoundingBoxes, self.object_detector_callback, queue_size=10)
        self.lane_detector = rospy.Subscriber(
            "lane_detector", Float32, self.lane_detector_callback, queue_size=10)
        print(self.lane_detector)

        self.motor_control = xycar_motor()
        self.motor_pub = rospy.Publisher(
            'xycar_motor', xycar_motor, queue_size=10)

        self.class_no = 0
        self.obj_bbox = None

        self.is_green = True
        self.stop_is_done = False

        self.angle = 0
        self.lpos = 0
        self.rpos = 640

    def object_detector_callback(self, msg):
        print("!!! object_detector_callback !!!")
        print('msg.data : {}'.format(msg))
        print('self : {}'.format(self))
        msg = msg.data[0]
        self.class_no = msg.class_no
        if self.class_no == MainControl.NO_OBJECT:
            self.obj_bbox = None
        else:
            self.obj_bbox = np.array(msg.bounding_box)

        if self.class_no == MainControl.TRAFFIC_LIGHT_SIGN:
            self.is_green = msg.is_green

        print('!!! class_no : {} / bounding_box : {}'.format(self.class_no, self.obj_bbox))

    def lane_detector_callback(self, msg):
        print("!!! lane_detector_callback !!!")
        print('msg.data : {}'.format(msg))
        print('self : {}'.format(self))

        self.angle = msg.angle
        #ㅠself.good_distance_to_stop = msg.good_distance_to_stop
        self.lpos = msg.lpos
        self.rpos = msg.rpos
        print('!!! angle : {:_<3} / lpos : {:_<3} / rpos :  {:_<3} / good_distance_to_stop : {}'.format(
            self.angle, self.lpos, self.rpos, self.good_distance_to_stop))

    def _go(self):
        self._pub(angle=self.angle)

    def _go_left(self):
        self.rpos = self.lpos + 470
        angle = self._get_angle(self.lpos, self.rpos)
        self._pub(angle=angle)

    def _go_right(self):
        self.lpos = self.rpos - 470
        angle = self._get_angle(self.lpos, self.rpos)
        self._pub(angle=angle)

    def _stop(self):
        self._pub(angle=0, speed=0)

    def _pub(self, angle=0, speed=3):
        self.motor_control.angle = angle * 0.4
        self.motor_control.speed = speed
        self.motor_pub.publish(self.motor_control)

    def _get_angle(self, l, r):
        return -(320 - (l + r) / 2) * 0.4

    def _drive(self):
        if not self.stop_is_done and self.class_no == MainControl.CROSSWALK_SIGN or self.class_no == MainControl.STOP_SIGN:
            self.status = 'normal'
            time.sleep(3)
            self._stop()
            time.sleep(5)
            self.class_no = MainControl.NO_OBJECT
            self.stop_is_done = True
        elif self.class_no == MainControl.LEFT_SIGN:
            self.ELAPSED_TIME = time.time()
            self.status = 'left'
            self.stop_is_done = False
        elif self.class_no == MainControl.RIGHT_SIGN:
            self.ELAPSED_TIME = time.time()
            self.status = 'right'
            self.stop_is_done = False
        elif self.class_no == MainControl.TRAFFIC_LIGHT_SIGN:
            self.status = 'normal'
            time.sleep(3)
            while not self.is_green:
                self._stop()
            self._go()
            self.stop_is_done = False

        if self.ELAPSED_TIME != -1 and (time.time() - self.ELAPSED_TIME > self.DURATION):
            self.status = 'normal'
            self.stop_is_done = False
            self.ELAPSED_TIME = -1

        elif self.status == 'right':
            self._go_right()
        elif self.status == 'left':
            self._go_left()

        if self.status == 'normal':
            self._go()

        if self.lpos == 0 or self.rpos == 0:
            self._go()

    def run(self):
        while True:
            self._drive()
            rospy.spin()


if __name__ == '__main__':
    print("!!!main")
    MainControl().run()
    print("!!!main2")
