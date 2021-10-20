import numpy as np
import cv2
import pytesseract
from typing import Tuple, Union
import math
import cv2
import numpy as np
import easyocr


def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

image = cv2.imread('images/5.jpg')
original = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([9, 89, 194], dtype="uint8")
upper = np.array([33, 255, 255], dtype="uint8")
mask = cv2.inRange(image, lower, upper)

cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(original, (x, y), (x + w, y + h), (0,0,255), 2)

mx = (0,0,0,0)      # biggest bounding box so far
mx_area = 0
for cont in cnts:
    x,y,w,h = cv2.boundingRect(cont)
    area = w*h
    if area > mx_area:
        mx = x,y,w,h
        mx_area = area
x,y,w,h = mx

print(mx_area)


cv2.fillPoly(mask, cnts, [255, 255, 255])
mask = 255 - mask
masked = cv2.bitwise_and(image,image,mask= mask)
cropped = original[y:y+h,x:x+w]


inverted = cv2.bitwise_not(cropped, cv2.COLOR_BGR2HSV)
inverted = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)
# (hMin = 0 , sMin = 0, vMin = 179), (hMax = 179 , sMax = 123, vMax = 255)
lower = np.array([0, 0, 100], dtype="uint8")
upper = np.array([150, 150, 255], dtype="uint8")
mask2 = cv2.inRange(inverted, lower, upper)
masked2 = cv2.bitwise_and(inverted,inverted,mask= mask2)


gray = cv2.cvtColor(masked2, cv2.COLOR_BGR2GRAY)
binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 1)
# binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

#morph ops
kernel = np.ones((2,2), np.uint8)
erosion = cv2.erode(binarized, kernel, iterations=1)
dilasion = cv2.dilate(binarized, kernel, iterations=1)


pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract"
print(pytesseract.image_to_string(dilasion, lang="eng"))



cv2.imwrite("numberplate.jpg", dilasion)
cv2.imwrite("original.jpg", original)
reader = easyocr.Reader(['en'])
result = reader.readtext("D:/UGRP-Truck-Number-Plate-Recognition/numberplate.jpg",paragraph="False")

print("\n\nEASYOCR:\n", result)
