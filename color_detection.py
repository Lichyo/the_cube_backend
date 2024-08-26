import numpy as np
import cv2
from PIL import ImageDraw, ImageFont
import json

# HSV values for the specified colors
white = [0, 0, 255]
yellow = [30, 255, 255]
red = [0, 255, 255]
orange = [15, 255, 255]
blue = [120, 255, 255]
green = [60, 255, 255]
color_list = ['orange', 'red', 'white', 'yellow', 'blue', 'green']


def init_user_define_colors(user):
    global white
    global yellow
    global red
    global orange
    global blue
    global green
    with open(f'user_define_colors/{user}.json', 'r') as f:
        user_define_colors = json.load(f)
        colors = user_define_colors['colors']
        white = colors["white"]
        yellow = colors["yellow"]
        red = colors["red"]
        orange = colors["orange"]
        blue = colors["blue"]
        green = colors["green"]
        for color in color_list:
            print(f"{color}: {get_color(color)}")


def get_color(color):
    if color == 'orange':
        return orange
    elif color == 'red':
        return red
    elif color == 'white':
        return white
    elif color == 'yellow':
        return yellow
    elif color == 'blue':
        return blue
    elif color == 'green':
        return green
    else:
        return [0, 0, 0]


def get_color_range(hsv):
    lower = []
    upper = []
    color_range = 30
    for i in range(3):
        lower.append(max(hsv[i] - color_range, 0))
        upper.append(min(hsv[i] + color_range, 255))
    return np.array(lower), np.array(upper)


def process_image(image, color, section_width, scan_area, records):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    color_in_hsv = get_color(color)
    lower, upper = get_color_range(color_in_hsv)
    output = cv2.inRange(image, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    output = cv2.dilate(output, kernel)
    output = cv2.erode(output, kernel)
    contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = find_section_range(scan_area, section_width)
    # for i in range(0, 9):
    #     x, y = points[i]
    #     print(f"Checking point: {i} : HSV: {image[y, x]}")
    # print('-------------------')

    for contour in contours:
        area = cv2.contourArea(contour)
        if (section_width * section_width * 0.7) < area:
            for i in range(0, 9):
                x, y = points[i]
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    cv2.circle(image, (x, y), 5, [0, 0, 0], -1)
                    records[i] = color
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image, records


def draw_3x3_grid(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    height, width, _ = image.shape

    color = (0, 0, 0)
    thickness = 4

    section_width = (width - 90) // 3
    section_height = section_width

    start_x = 45
    end_x = width - 45
    start_y = int(height / 2 - section_width * 1.5)
    end_y = int(height / 2 + section_width * 1.5)

    for i in range(0, 4):
        x = i * section_width + start_x
        cv2.line(image, (x, start_y), (x, end_y), color, thickness)

    for i in range(0, 4):
        y = start_y + i * section_height
        cv2.line(image, (start_x, y), (end_x, y), color, thickness)

    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    scan_area = (start_x, start_y, end_x, end_y)
    center_points = (int(start_x + 1.5 * section_width), int(start_y + 1.5 * section_height))
    return image, section_width, scan_area, center_points


def draw_banner(image):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = "Cy_Cube"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    position = (image.width - text_width - 10, 10)
    draw.text(position, text, font=font, fill="white")
    return image


def find_section_range(scan_area, section_width):
    start_x, start_y, end_x, end_y = scan_area
    offset_x = (end_x - section_width // 2)
    offset_y = (end_y - section_width // 2)
    records = []
    for i in range(0, 3):
        for j in range(2, -1, -1):
            x = offset_x - j * section_width
            y = offset_y - i * section_width
            records.append((x, y))
    return records


def get_center_color_hsv(image, center_points):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    x, y = center_points
    return image[y, x]
