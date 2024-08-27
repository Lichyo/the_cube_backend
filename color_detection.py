import numpy as np
import cv2
from PIL import ImageDraw, ImageFont
import json

# HSV values for the specified colors
color_list = ['orange', 'red', 'white', 'yellow', 'blue', 'green']


def get_user_define_colors(user):
    with open(f'user_define_colors/{user}.json', 'r') as f:
        user_define_colors = json.load(f)
        return user_define_colors


def get_color_range(rgb):
    lower = []
    upper = []
    color_range = 20
    for i in range(3):
        lower.append(max(rgb[i] - color_range, 0))
        upper.append(min(rgb[i] + color_range, 255))
    return np.array(lower), np.array(upper)


def process_image(image, color, section_width, scan_area, records):
    image = cv2.GaussianBlur(np.array(image), (5, 5), 0)
    user_define_colors = get_user_define_colors('chiyu')
    user_define_colors = user_define_colors["colors"]
    image = np.array(image)

    points = find_section_range(scan_area, section_width)
    for i in range(0, 9):
        x, y = points[i]
        contrast = 0
        brightness = 10 - i // 3 * 8
        output = image * (contrast / 127 + 1) - contrast + brightness  # 轉換公式
        output = np.clip(output, 0, 255)
        image = np.uint8(output)

        color_in_rgb = user_define_colors[color]
        lower, upper = get_color_range(color_in_rgb)
        output = cv2.inRange(image, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        output = cv2.dilate(output, kernel)
        output = cv2.erode(output, kernel)
        contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                records[i] = color
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


def get_center_color_rgb(image, center_points):
    image = np.array(image)
    x, y = center_points
    return image[y, x]
