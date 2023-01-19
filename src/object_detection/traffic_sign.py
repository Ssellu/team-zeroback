import cv2
import numpy as np


class TrafficLight():
    def __init__(self, is_debug: bool = False) -> None:
        self.is_debug = is_debug

    def is_green(self, path: str, xywh: np.ndarray) -> bool:
        image, opencv_xywh = self.crop_image_blurred_(path, xywh)

        _, _, v = self._get_hsv_(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, 1, 20, param1=25, param2=25, minRadius=0, maxRadius=30)

        # Exception
        if circles is None:
            raise AssertionError(
                'The traffic light exists but it\'s ignorable.')

        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            # [center_x, center_y, radius]
            cv2.circle(image, (circle[0], circle[1]),
                       circle[2], (0, 255, 0), 2)
            cv2.circle(image, (circle[0], circle[1]), 2, (0, 0, 255), 3)

            cr_img = v[circle[1]-circle[2]: circle[1]+circle[2],
                       circle[0]-circle[2]: circle[0]+circle[2]]

            # opencv_xywh[3] : total height of cropped image
            if cr_img.mean() > 200 and circle[1] > opencv_xywh[3] * 0.666:
                return True

        if self.is_debug:
            print("!!! len(circles) : {}".format(len(circles)))
            print("!!! len(circles[0, :]) : {}".format(len(circles[0, :])))
            print("!!! radius : {}".format(circle[2]))
            print("!!! opencv_xywh : {}".format(opencv_xywh))
            cv2.imshow('circles', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return False

    def crop_image_blurred_(self, path, xywh):
        # for detect circles
        min_x, min_y, width, height = xywh

        min_x_640 = int(min_x * 640)
        min_y_480 = int(min_y * 480)
        width_640 = int(width * 640)
        height_480 = int(height * 480)

        start_x = min_x_640 - width_640//2
        start_y = min_y_480 - height_480//2

        image = cv2.imread(path)[start_y:start_y +
                                 height_480, start_x:start_x+width_640]
        image = cv2.GaussianBlur(image, (5, 5), 0)
        return image, (start_x, start_y, width_640, height_480)

    def _get_hsv_(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if self.is_debug:
            cv2.imshow('v', v)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return h, s, v


class StopLine():

    def __init__(self, is_debug: bool = False, roi=None) -> None:
        self.is_debug = is_debug
        self.calibration_matrix = np.array([[422.037858, 0.0, 245.895397], [0.0, 435.5899734, 163.625535], [0.0, 0.0, 1.0]])
        self.calibration_coefficients = np.array([-0.302321,0.071616, -0.002477, -0.000052, 0.00000])
        self.roi = roi
        self.hstack_cat = []

    def white_(self, image=None, path=None, roi=None):
        original = image if image is not None else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        h, l, s = cv2.split(cv2.cvtColor(original, cv2.COLOR_BGR2HLS))
        _, h = cv2.threshold(h, 125, 255, cv2.THRESH_BINARY_INV)
        _, l = cv2.threshold(l, 125, 255, cv2.THRESH_BINARY_INV)
        _, s = cv2.threshold(s, 125, 255, cv2.THRESH_BINARY_INV)

        if self.is_debug and False:
            cv2.imshow('original', original)
            cv2.imshow('H', h)
            cv2.imshow('L', l)
            cv2.imshow('S', s)
            cv2.waitKey()
            cv2.destroyAllWindows()
        return l

    def calibrate_image_(self, frame, mtx, dist, cal_mtx, cal_roi):
        tf_image = cv2.undistort(frame, mtx, dist, None, cal_mtx)
        x, y, w, h = cal_roi
        tf_image = tf_image[y:y+h, x:x+w]
        return cv2.resize(tf_image, (frame.shape[1], frame.shape[0]))

    def is_stopline_in_roi(self, path: str = None, roi=None):
        image_l = self.white_(path=path, roi=roi)
        cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(
            self.calibration_matrix,
            self.calibration_coefficients,
            (480, 640),
            1,
            (480, 640)
        )


        if self.is_debug:
            cal_image = self.calibrate_image_(
            image_l,
            self.calibration_matrix,
            self.calibration_coefficients,
            cal_mtx,
            cal_roi
            )
            self.hstack_cat = np.hstack((image_l, cal_image))
            meet_stopline = self.detect_stopline_(cal_image=cal_image, low_threshold_value=150)
            print('!!! meet_stopline : {}'.format(meet_stopline))
            cv2.imshow('calibration', cal_image)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def detect_stopline_(self, cal_image, low_threshold_value) -> bool:
        stopline_roi, _, _ = self.set_roi_(cal_image, 250, 330, 10)
        image = self.image_processing_(stopline_roi, low_threshold_value)
        if cv2.countNonZero(image) > 1000:
            return True
        return False

    def set_roi_(self, frame, x_len, start_y, offset_y):
        _, width = frame.shape
        start_x = int(width/2 - (x_len/2))
        end_x = int(width - start_x)
        return frame[start_y:start_y + offset_y, start_x:end_x], start_x, start_y

    def image_processing_(self, image, low_threshold_value):
        _, lane = cv2.threshold(image, low_threshold_value, 255, cv2.THRESH_BINARY)
        return lane

if __name__ == '__main__':

    if False:
        # Test Case 1
        ts = np.array([0.201784, 0.274947, 0.198439, 0.481953])
        tl = TrafficLight(is_debug=True)
        print('result : {}'.format(tl.is_green(
            path='dataset/image_sets/img (141).png', xywh=ts)))
    if True:
        sl = StopLine(is_debug=True)
        sl.is_stopline_in_roi(path='dataset/image_sets/img (141).png')
