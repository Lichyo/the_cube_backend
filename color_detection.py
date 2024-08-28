import numpy as np
import cv2
from PIL import ImageDraw, ImageFont
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

color_list = ['orange', 'red', 'white', 'yellow', 'blue', 'green']


def predict_color(image, section_width, scan_area, user):
    points = find_center_points(scan_area, section_width)
    classifier, acc = get_classifier(user)
    print(f"""Accuracy: {acc}""")
    records = []
    for i in range(0, 9):
        x, y = points[i]
        source = image[y, x]
        color = classifier.predict([source])
        records.append(color)
    return records


def get_classifier(user):
    data = pd.read_csv('user_define_colors/' + user + '.csv')
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    sc_x = StandardScaler()

    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)

    classifier = SVC(kernel='rbf')
    classifier.fit(x_train, y_train)
    return classifier, accuracy_score(y_test, classifier.predict(x_test))


def init_color_dataset(user, color, image, section_width, scan_area):
    image = np.array(image)
    points = find_center_points(scan_area, section_width)
    colors = []

    file_path = f'user_define_colors/{user}.csv'
    try:
        existing_df = pd.read_csv(file_path)
    except FileNotFoundError:
        existing_df = pd.DataFrame(columns=['R', 'G', 'B', 'Color'])

    for (x, y) in points:
        source = image[y, x]
        colors.append((source[0], source[1], source[2], color))

    # Create new DataFrame
    new_df = pd.DataFrame(colors, columns=['R', 'G', 'B', 'Color'])

    # Append new data to existing data
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Save updated data to CSV
    updated_df.to_csv(file_path, index=False)
    print(f"Dataset for {user} has been updated")


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


def find_center_points(scan_area, section_width):
    start_x, start_y, end_x, end_y = scan_area
    offset_x = (end_x - section_width // 2)
    offset_y = (end_y - section_width // 2)
    records = []
    for i in range(0, 3):
        for j in range(0, 3):
            x = offset_x - j * section_width
            y = offset_y - i * section_width
            records.append((x, y))
    return records
