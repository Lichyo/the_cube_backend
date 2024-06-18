from flask import Flask, request, Response
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)


@app.route('/')
def hello_world():
    print('service called')
    return 'Hello, World!'


@app.route('/process_image', methods=['GET'])
def process_frame():
    # image = request.args.get('image')
    print('service called')
    # try:
    #     # Decode the base64 image
    #     image_data = base64.b64decode(image)
    #     np_arr = np.frombuffer(image_data, np.uint8)
    #     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    #     if img is None:
    #         return "Image decoding failed", 400
    #
    #     # Process the image using OpenCV (convert to grayscale)
    #     processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    #     # Save the processed image as a JPG file
    #     output_filename = 'processed_image.jpg'
    #     cv2.imwrite(output_filename, processed_img)
    #
    #     return "Image saved successfully", 200
    # except Exception as e:
    #     return str(e), 400


if __name__ == '__main__':
    # Ensure the output directory exists
    # os.makedirs('output', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
