import numpy as np
import cv2

orange = [100, 100, 230]
red = [54, 54, 187]
white = [181, 193, 183]
yellow = [79, 202, 181]
blue = [99, 99, 44]
green = [159, 159, 12]


def process_image(img, color):
    lower, upper = get_color_range(color)
    output = cv2.inRange(img, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    output = cv2.dilate(output, kernel)
    output = cv2.erode(output, kernel)
    contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        # color = get_outline_color(color)
        if 4000 > area > 3000:
            x, y, w, h = cv2.boundingRect(contour)  # 取得座標與長寬尺寸
            img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)  # 繪製四邊形
    return img


def get_outline_color(color):
    if color == orange:
        return color(0, 150, 255)
    elif color == red:
        return color(0, 0, 255)
    elif color == white:
        return color(255, 255, 255)
    elif color == yellow:
        return color(0, 255, 255)
    elif color == green:
        return color(0, 255, 0)
    else:
        return color(255, 0, 0)


def get_color_range(bgr):
    lower = []
    upper = []
    for i in range(3):
        if bgr[i] - 40 > 0:
            lower.append(bgr[i] - 35)
        else:
            lower.append(0)
    for i in range(3):
        if bgr[i] + 40 < 255:
            upper.append(bgr[i] + 35)
        else:
            upper.append(255)
    lower = np.array(lower)
    upper = np.array(upper)
    return lower, upper
