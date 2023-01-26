from object_detection.traffic_sign import *
import time
import rospy
from std_msgs.msg import Float32
from object_detection.traffic_sign import *

from xycar_msgs.msg import xycar_motor

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

    def __init__(self) -> None:

        rospy.init_node("Control",anonymous=False)

        self.object_detector = rospy.Subscriber("object_detector", Float32, self.object_detector_callback, queue_size = 10)
        self.lane_detector = rospy.Subscriber("lane_detector", Float32, self.lane_detector_callback, queue_size = 10)

        self.motor_control = xycar_motor()
        self.motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=10)

        self.class_no = 0
        self.obj_bbox = None

        self.is_green = True

    def object_detector_callback(self, msg):
        print("!!! object_detector_callback !!!")
        print('msg.data : {}'.format(msg))
        print('self : {}'.format(self))
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
        self.good_distance_to_stop = msg.good_distance_to_stop
        self.lpos = msg.lpos
        self.rpos = msg.rpos
        print('!!! angle : {:_<3} / lpos : {:_<3} / rpos :  {:_<3} / good_distance_to_stop : {}'.format(self.angle, self.lpos, self.rpos, self.good_distance_to_stop))


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
        self.motor_control.angle = angle
        self.motor_control.speed = speed
        self.motor_pub.publish(self.motor_control)

    def _get_angle(l, r):
        return -(320 - (l + r) / 2) * 0.4

    def _drive(self):
        if self.class_no == MainControl.CROSSWALK_SIGN or self.class_no == MainControl.STOP_SIGN:
            if self.good_distance_to_stop:
                self._stop()
        elif self.class_no == MainControl.LEFT_SIGN:
            self._go_left()
        elif self.class_no == MainControl.RIGHT_SIGN:
            self._go_right()
        elif self.class_no == MainControl.TRAFFIC_LIGHT_SIGN:
            while not self.is_green:
                self._stop()
        self._go()

    def run(self):
        while True:
            self._drive()
            rospy.spin()


if __name__ == '__main__':

    MainControl().run()

