import fitz
import cv2
import pandas as pd
import os
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
import math
from typing import Tuple, Union
import json
from deskew import determine_skew
from pytesseract import Output
import threading
import imageio


def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + \
        abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + \
        abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def imagePreprocessing(image):
    scale_factor = 4000 / image.shape[0]  # percent of original size
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(resized.shape, dtype=np.uint8)

    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cv2.fillPoly(mask, cnts, [255, 255, 255])
    mask = 255 - mask
    masked = cv2.bitwise_or(resized, mask)

    grayscale = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    deskewed = rotate(grayscale, angle, (0, 0, 0))
    inverted = cv2.bitwise_not(deskewed, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(inverted, kernel, iterations=2)
    erosion = cv2.bitwise_not(erosion)
    denoised = cv2.fastNlMeansDenoising(erosion, None, 20, 31, 7)
    # binarized = cv2.adaptiveThreshold(
    #   erosion, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 1)
    binarized = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # plt.imshow(binarized, "gray")
    # plt.show()
    return binarized


def arraySplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def OCR(binarizedImg, data, fname, outputJSON):
    data.append({
        "filename": fname,
        "data": pytesseract.image_to_data(binarizedImg, output_type=Output.DICT, lang="eng"),
        "text": pytesseract.image_to_string(binarizedImg, lang="eng"),
        "boxes": pytesseract.image_to_boxes(binarizedImg, lang="eng")
    })
    with open(outputJSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def batchOCR(inputDirectory, outputJSON):
    with open("gif-to-text.json", 'r', encoding='utf-8') as f:
        compare = json.load(f)
    data = []
    for subdir, dirs, files in os.walk(inputDirectory):
        for file in files:
            filepath = subdir + os.sep + file
            filename = filepath.rsplit("\\", 1)[1]
            print(filename)
            if not any(i['filename'] == filename for i in compare):
                try:
                    if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                        image = cv2.imread(filepath)
                        binarizedImg = imagePreprocessing(image)
                        OCR(binarizedImg, data, filepath, outputJSON)
                    if filepath.lower().endswith(('.gif')):
                        image = imageio.mimread(filepath)[0]
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        if not (image.shape[0] * image.shape[1]) < 120000:
                            binarizedImg = imagePreprocessing(image)
                            OCR(binarizedImg, data, filepath, outputJSON)
                        else:
                            print("image to small:", filepath)
                            data.append({
                                "filename": filepath,
                                "tooSmall": True
                            })
                            with open(outputJSON, 'w', encoding='utf-8') as f:
                                json.dump(
                                    data, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    print("error in OCRing file: ", filename, "\n", e)


batchOCR("G:\\OCR\\gif", 'G:\\OCR\\gif-to-text2.json')
