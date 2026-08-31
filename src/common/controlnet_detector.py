import cv2
import numpy as np
from PIL import Image
from .guided_filter import FastGuidedFilter
from .util import HWC3

UPSCALE_METHODS = ["INTER_NEAREST", "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"]


def get_upscale_method(method_str):
    if method_str not in UPSCALE_METHODS:
        raise ValueError(f"Method {method_str} not found in {UPSCALE_METHODS}")
    return getattr(cv2, method_str)


def apply_gaussian_blur(image_np, ksize=5, sigmaX=1.0):
    if ksize % 2 == 0:
        ksize += 1
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=sigmaX)
    return blurred_image


def apply_guided_filter(image_np, radius, eps, scale):
    guided_filter = FastGuidedFilter(image_np, radius, eps, scale)
    return guided_filter.filter(image_np)


def canny_detector(image: Image, low_threshold=50, high_threshold=100):
    image = np.array(image)
    canny_img = cv2.Canny(image, low_threshold, high_threshold)
    canny_img = HWC3(canny_img)
    return Image.fromarray(canny_img)


def tile_detector(image: Image, mode="blur", blur_strength=7, radius=5, eps=0.4, scale_factor=4):
    image = np.array(image)
    tile_map = image
    if mode == "blur":
        blur_img = apply_gaussian_blur(image, ksize=int(blur_strength), sigmaX=blur_strength / 2)
        blur_img = apply_guided_filter(blur_img, radius, eps, scale_factor)
        tile_map = blur_img
    return Image.fromarray(tile_map)
