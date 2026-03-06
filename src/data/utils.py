import cv2
import hdf5storage as h5
import numpy as np
import random

def load_image(path: str):
    """ Loads an image from the specified path and converts it to RGB format. """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_mat(path: str, key: str):
    """ Loads a backward mapping from the specified path. """
    bm = h5.loadmat(path)[key]
    return bm


def crop_image_tight(img, bm):
    """
    Crops the image tightly around using the backward mapping.
    This function creates a tight crop around the document in the image.
    """
    size = img.shape

    minx = np.floor(np.amin(bm[:, :, 0])).astype(int)
    maxx = np.ceil(np.amax(bm[:, :, 0])).astype(int)
    miny = np.floor(np.amin(bm[:, :, 1])).astype(int)
    maxy = np.ceil(np.amax(bm[:, :, 1])).astype(int)
    s = 20
    s = min(min(s, minx), miny)  # s shouldn't be smaller than actually available natural padding is
    s = min(min(s, size[1] - 1 - maxx), size[0] - 1 - maxy)

    # Crop the image slightly larger than necessary
    img = img[miny - s : maxy + s, minx - s : maxx + s, :]
    cx1 = random.randint(0, max(s - 5, 1))
    cx2 = random.randint(0, max(s - 5, 1)) + 1
    cy1 = random.randint(0, max(s - 5, 1))
    cy2 = random.randint(0, max(s - 5, 1)) + 1

    img = img[cy1:-cy2, cx1:-cx2, :]
    top = miny - s + cy1
    bot = size[0] - maxy - s + cy2
    left = minx - s + cx1
    right = size[1] - maxx - s + cx2
    return img, top, bot, left, right

def crop_tight(img, bm):
    H, W = img.shape[:2]
    img, top, bot, left, right = crop_image_tight(img, bm)
    bm[:, :, 0] -= left
    bm[:, :, 1] -= top
    bm = bm / np.array([W - left - right, H - top - bot])
    return img / 255.0, bm * 2 - 1  # Scale BM to [-1, 1]   