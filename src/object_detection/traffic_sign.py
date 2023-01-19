import cv2
import numpy as np


class TrafficLight():
    def __init__(self, is_debug: bool = False) -> None:
        self.is_debug = is_debug
        self.v = None

    def is_green(self, path: str, xywh: np.ndarray) -> bool:
        image, opencv_xywh = self.crop_image_blurred(path, xywh)

        _, _, v = self._get_hsv_(image)
        print(v)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, 1, 20, param1=25, param2=25, minRadius=0, maxRadius=30)
        # Exception
        if circles is None:
            raise AssertionError('The traffic light exists but it\'s ignorable.')


        circles = np.uint16(np.around(circles))

        print("!!! len(circles) : {}".format(len(circles)))
        print("!!! len(circles[0, :]) : {}".format(len(circles[0, :])))
        for circle in circles[0, :]:
            # [center_x, center_y, radius]
            cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            cv2.circle(image, (circle[0], circle[1]), 2, (0, 0, 255), 3)

            print("!!! radius : {}".format(circle[2]))
            cr_img = v[circle[1]-circle[2] : circle[1]+circle[2],
                       circle[0]-circle[2] : circle[0]+circle[2]]

            # opencv_xywh[3] : total height of cropped image
            print("!!! opencv_xywh : {}".format(opencv_xywh))
            if cr_img.mean() > 200 and circle[1] > opencv_xywh[3] * 0.666:
                return True

        if self.is_debug:
            cv2.imshow('circles', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return False

    def crop_image_blurred(self, path, xywh):
        # for detect circles
        min_x, min_y, width, height = xywh

        min_x_640 = int(min_x * 640)
        min_y_480 = int(min_y * 480)
        width_640 = int(width * 640)
        height_480 = int(height * 480)

        start_x = min_x_640 - width_640//2
        start_y = min_y_480 - height_480//2

        image = cv2.imread(path)[start_y:start_y+height_480, start_x:start_x+width_640]
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


if __name__ == '__main__':

    # Test Case 1
    ts = np.array([0.201784, 0.274947, 0.198439, 0.481953])


    tl = TrafficLight(is_debug=True)
    print('result : {}'.format(tl.is_green(path='dataset/image_sets/img (141).png', xywh=ts)))
