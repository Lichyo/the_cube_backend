import numpy as np
import cv2
from PIL import ImageDraw, ImageFont

orange = [38, 112, 230]  # ok
red = [17, 15, 164]  # ok
white = [181, 193, 183]  # ok
yellow = [35, 185, 157]  # ok
blue = [149, 71, 11]  # ok
green = [10, 145, 10]  # ok
color_list = [white, yellow, orange, red, blue, green]


def process_image(image, color, section_width, scan_area, records):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lower, upper = get_color_range(color)
    output = cv2.inRange(image, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    output = cv2.dilate(output, kernel)
    output = cv2.erode(output, kernel)
    contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if (section_width * section_width * 0.7) < area:
            x, y, w, h = cv2.boundingRect(contour)  # 取得座標與長寬尺寸
            print('x:', x, 'y:', y, 'w:', w, 'h:', h, 'area:', area)
            # image = cv2.drawContours(image, contours, -1, color, thickness=3)
            points = find_section_range(scan_area, section_width)
            for i in range(0, 9):
                x, y = points[i]
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    cv2.circle(image, (x, y), 5, [0, 0, 0], -1)
                    records[i] = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, records


def get_color_range(bgr):
    lower = []
    upper = []
    for i in range(3):
        if bgr[i] - 40 > 0:
            lower.append(bgr[i] - 40)
        else:
            lower.append(0)
    for i in range(3):
        if bgr[i] + 40 < 255:
            upper.append(bgr[i] + 40)
        else:
            upper.append(255)
    lower = np.array(lower)
    upper = np.array(upper)
    return lower, upper


def draw_3x3_grid(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
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

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scan_area = (start_x, start_y, end_x, end_y)
    return image, section_width, scan_area


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
    offset_x = (start_x + section_width // 2)
    offset_y = (end_y - section_width // 2)
    records = []
    for i in range(0, 3):
        for j in range(0, 3):
            x = offset_x + j * section_width
            y = offset_y - i * section_width
            records.append((x, y))
    return records
