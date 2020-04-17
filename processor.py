"""
Process an image that we can pass to our networks.
"""
import cv2
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

def process_image(image, target_shape):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)

    return x

def process_flow(image1, image2, target_shape):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, _ = target_shape
    image1 = load_img(image1, target_size=(h, w))
    image2 = load_img(image2, target_size=(h, w))
    img_arr = img_to_array(image1)
    x1 = (img_arr / 255.).astype(np.float32)
    img_arr = img_to_array(image2)
    x2 = (img_arr / 255.).astype(np.float32)
    image1 = cv2.cvtColor(x1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(x2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow
