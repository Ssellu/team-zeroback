import cv2
import numpy as np


class TrafficLight():
    def __init__(self, is_debug: bool = False) -> None:
        self.is_debug = is_debug
        self.v = None

    def is_green(self, path: str, xywh: np.ndarray) -> bool:
        print('!!!is_green')
        image, *opencv_xywh = self.crop_image_(path, xywh)
        return bool(self.get_circles_(image, opencv_xywh)[2])

    def get_circles_(self, image, opencv_xywh):

        _, _, v = self._get_hsv_(image)
        cimg = cv2.GaussianBlur(image, (5, 5), 0)

        print(image.type() == cv2.CV_8UC1)
        print((image.isMat() or image.isUMat()))
        #scimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=25, minRadius=0, maxRadius=30)
        circles = np.uint16(np.around(circles))
        circle = circles[0, -1]
        cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)

        cr_img = v[circle[1]-10: circle[1]+10, circle[0]+10]
        img_str = 'x : {}, y : {}, mean : {}'.format(circle[0], circle[1], cr_img.mean())
        print(img_str)
        for circle in circles[0, :]:
            # [center_x, center_y, radius]
            cv2.circle(cimg, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            cv2.circle(cimg, (circle[0], circle[1]), 2, (0, 0, 255), 3)

            cr_img = v[circle[1]-10: circle[1]+10, circle[0]+10]
            img_str = 'x : {}, y : {}, mean : {}'.format(
                circle[0], circle[1], cr_img.mean())
            print(img_str)

        if self.is_debug:
            cv2.imshow('circles', cimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def crop_image_(self, path, xywh):
        # for detect circles
        min_x, min_y, width, height = xywh

        min_x_640 = int(min_x * 640)
        min_y_480 = int(min_y * 480)
        width_640 = int(width * 640)
        height_480 = int(height * 480)

        start_x = min_x_640 - width_640//2
        start_y = min_y_480 - height_480//2

        image = cv2.imread(path)[start_y:start_y+height_480, start_x:start_x+width_640]
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
    ts = np.array([0.701784, 0.383495, 0.063545, 0.130097])

    # Test Case 2
    # ts = np.array([0.963768, 0.355340, 0.065775, 0.217476])

    tl = TrafficLight(is_debug=True)
    print(tl.is_green(path='dataset/image_sets/img (3).png', xywh=ts))
