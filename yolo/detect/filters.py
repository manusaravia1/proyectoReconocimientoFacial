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


def contrast(image):
    new_image = image.copy()
    coefficient = 0.8

    w, h, c = image.shape

    avg = 0
    for x in range(w):
        for y in range(h):
            r, g, b = image[x, y]
            avg += (float(r / 3) + float(g / 3) + float(b / 3))

    avg /= w * h

    palette = []
    for i in range(256):

        if i > 180:
            coefficient = (255 - i) * 0.0066 + 0.5
        else:
            coefficient = 1

        if avg > 170 and i < 100:
            coefficient = 1.5

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
    # Open an already existing image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)

    return sharpened


def align(image):  # NO APLICAR

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    desiredLeftEye = (0.35, 0.35)
    desiredFaceWidth = 256
    desiredFaceHeight = 256

    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rect = detector(gray, 2)

    if (len(rect) > 0):
        rect = rect[0]

    (x, y, w, h) = rect_to_bb(rect)
    faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)

    shape = predictor(gray, rect)

    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    shape = coords

    # faceAligned = fa.align(image, gray, rect)
    # display the output images
    # cv2.imshow("Original", faceOrig)
    # cv2.imshow("Aligned", faceAligned)
    # cv2.waitKey(0)

    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    desiredRightEyeX = 1.0 - desiredLeftEye[0]
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2, (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    return output
