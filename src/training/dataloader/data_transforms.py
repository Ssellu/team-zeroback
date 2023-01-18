from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms as tf

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# from utils.tools import xywh2xyxy_np


def get_transformations(cfg_param=None, is_train=None):
    if is_train:
        # Make augmentated images into a bunch
        data_transform = tf.Compose([AbsoluteLabels(),
                                     DefaultAug(),
                                     RelativeLabels(),
                                     ResizeImage(
                                         new_size=(cfg_param['width'], cfg_param['height'])),
                                     ToTensor(),
                                     ])
    else:
        data_transform = tf.Compose([AbsoluteLabels(),
                                     RelativeLabels(),
                                     ToTensor(),])

    return data_transform


# Convert normalized values in bounding boxes to absolute value
class AbsoluteLabels(object):
    def __init__(self) -> None:
        pass

    def __call__(self, data):
        image, label = data
        h, w, _ = image.shape
        label[:, [1, 3]] *= w  # cx * w
        label[:, [2, 4]] *= h  # cy * h
        return image, label

# Convert normalized values in bounding boxes to relative value


class RelativeLabels(object):

    def __init__(self) -> None:
        pass

    def __call__(self, data):
        image, label = data
        h, w, _ = image.shape
        label[:, [1, 3]] /= w  # cx / w
        label[:, [2, 4]] /= h  # cy / h
        return image, label


# Augmentation Template class
class ImageAug(object):
    def __init__(self, augmentations=[]) -> None:
        self.augmentations = augmentations

    def __call__(self, data):
        # Unpack data
        img, labels = data

        # Convert xywh -> xyxy (minx, miny, maxx, maxy)
        boxes = np.array(labels)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

        # Convert bounding box to imgaug format
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes],
            shape=img.shape
        )

        # Apply augmentations
        img, bounding_boxes = self.augmentations(image=img,
                                                 bounding_boxes=bounding_boxes)

        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding box to np.ndarray
        boxes = np.zeros((len(bounding_boxes), 5))

        for box_idx, box in enumerate(bounding_boxes):
            x1, y1, x2, y2 = box.x1, box.x2, box.y1, box.y2

            # Reshape back to [x,y,w,h]
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = (x1 + x2) / 2
            boxes[box_idx, 2] = (y1 + y2) / 2
            boxes[box_idx, 3] = x2 - x1
            boxes[box_idx, 4] = y2 - y1
        return img, boxes

# Default Augmentation


class DefaultAug(ImageAug):
    def __init__(self) -> None:
        pass


"""
        iaa_list = [
            iaa.Add(value=25),
            iaa.Add(value=45),
            iaa.Add(value=-25),
            iaa.Add(value=-45),
        ] + [
            iaa.AdditiveGaussianNoise(scale=0.03*255),
            iaa.AdditiveGaussianNoise(scale=0.05*255),
            iaa.AdditiveGaussianNoise(scale=0.10*255),
            iaa.AdditiveGaussianNoise(scale=0.20*255),
        ] + [
            iaa.SaltAndPepper(p=0.01),
            iaa.SaltAndPepper(p=0.02),
            iaa.SaltAndPepper(p=0.03),
        ] + [
            iaa.Cartoon()
        ] + [
            iaa.BlendAlpha(factor=0.2)
        ] + [
GaussianBlur(sigma=0.25)
GaussianBlur(sigma=0.50)
GaussianBlur(sigma=1.00)
        ] + [
MotionBlur(0)
MotionBlur(72)
MotionBlur(144)
MotionBlur(216)
MotionBlur(288)
        ] + [
            ChangeColorTemperature(kelvin=8000)
ChangeColorTemperature(kelvin=16000)
        ] + [
RemoveSaturation(mul=0.25)
        ] + [
GammaContrast(gamma=0.50)
GammaContrast(gamma=0.81)
GammaContrast(gamma=1.12)
GammaContrast(gamma=1.44)
        ] + [
SigmoidContrast(gain=5.1)
SigmoidContrast(gain=17.1)
SigmoidContrast(gain=14.4)
        ] + [
HistogramEqualization(to_colorspace=HSV)
HistogramEqualization(to_colorspace=HLS)
HistogramEqualization(to_colorspace=Lab)
        ] + [
Sharpen(alpha=1, lightness=1.5)
Sharpen(alpha=1, lightness=1.2)
Sharpen(alpha=1, lightness=0.5)
Sharpen(alpha=1, lightness=0.8)
        ] + [
Emboss(alpha=1, strength=0.2)
Emboss(alpha=1, strength=0.3)
Emboss(alpha=1, strength=0.4)
        ] + [
ZoomBlur(severity=1)
ZoomBlur(severity=2)
ZoomBlur(severity=3)
ZoomBlur(severity=4)
ZoomBlur(severity=5)
        ] + [

        ] + [

        ] + [

        ] + [

        ] + [] + [] + [] + [] + [] + []


        self.augmentations = iaa.Sequential(iaa_list)

"""


class ResizeImage(object):
    def __init__(self, new_size, interpolation=cv2.INTER_LINEAR):
        self.new_size = tuple(new_size)
        self.interpolation = interpolation

    def __call__(self, data):
        image, label = data
        image = cv2.resize(image, self.new_size,
                           interpolation=self.interpolation)
        return image, label


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, data):
        image, labels = data
        image = torch.tensor(
            np.transpose(
                # Image Normalize
                np.array(image, dtype=float)/255., (2, 0, 1)
            ),
            dtype=torch.float32
        )  # H, W, C -> C, H, W
        labels = torch.FloatTensor(np.array(labels))

        return image, labels


if __name__ == '__main__':
    image = np.asarray(Image.open('img (194).png'))
    with open('img (194).txt', 'r') as f:
        bb_arr = [[float(n.replace('\n', ''))
                   for n in a_line.split(' ')[1:]]for a_line in f.readlines()]

        bbs = BoundingBoxesOnImage([
            BoundingBox(
                x1=((n[0] - n[2]/2)*image.shape[1]),
                x2=(n[0] + n[2]/2)*image.shape[1],
                y1=(n[1] - n[3]/2)* image.shape[0],
                y2=(n[1] + n[3]/2)*image.shape[0]
                ) for n in bb_arr
        ], shape=image.shape)

    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.Affine(
            translate_px={"x": 40, "y": 60},
            scale=(0.5, 0.7)
        ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
        iaa.SaltAndPepper(p=0.03)
    ])
    # seq = iaa.Identity()
    # Augment BBs and images.
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    # print coordinates before/after augmentation (see below)
    # use .x1_int, .y_int, ... to get integer coordinates
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
            i,
            before.x1, before.y1, before.x2, before.y2,
            after.x1, after.y1, after.x2, after.y2)
        )

    # image with BBs before/after augmentation (shown below)
    image_before = bbs.draw_on_image(image, size=2)
    image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
    print(image_aug.shape, bbs_aug.shape)
    im = Image.fromarray(image_after)
    im.save("your_file.png")
    print("444")
