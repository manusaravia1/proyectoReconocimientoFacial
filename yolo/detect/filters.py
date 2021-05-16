# from .helpers import FACIAL_LANDMARKS_IDXS
# from .helpers import shape_to_np
import numpy as np
import argparse
import imutils
import dlib
import cv2

from PIL import Image
from PIL import ImageFilter

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

'''
def brightness(image):
    new_image = image.copy()
    coefficient = 50

    h, w, c = image.shape

    for x in range(w):
        for y in range(h):
            for c in range(3):
                new_image[x,y][c] += coefficient
                if new_image[x,y][c] > 255:
                    new_image[x, y][c] = 255

    return new_image


def contrast(image):
    new_image = image.copy()
    coefficient = 1.2

    h, w, c = image.shape

    avg = 0
    for x in range(w):
        for y in range(h):
            r, g, b = image[x, y]
            avg += (float(r / 3) + float(g / 3) + float(b / 3))

    avg /= w * h
    palette = []
    for i in range(256):

        temp = int(avg + coefficient * (i - avg))
        if temp < 0:
            temp = 0
        elif temp > 255:
            temp = 255
        palette.append(temp)

    for x in range(w):
        for y in range(h):
            r, g, b = image[x, y]
            new_image[x, y] = (palette[r], palette[g], palette[b])

    return new_image


def sharp(image):

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)

    return sharpened

def blur(image):

    kernel = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
    blurred = cv2.filter2D(image, -1, kernel)

    return blurred
'''
'''
def align(image):  # NO APLICAR

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    desiredLeftEye = (0.35, 0.35)
    desiredFaceWidth = 256
    desiredFaceHeight = 256

    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)'''
