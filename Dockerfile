FROM python:3.10-slim-buster

WORKDIR /app

RUN pip install Flask==2.2.2
RUN pip install Flask-SocketIO==5.3.0
RUN pip install Pillow==9.3.0
RUN pip install scikit-learn==1.2.2
RUN pip install numpy==1.21.3
RUN pip install pandas==1.3.4

RUN pip wheel opencv-python==4.7.0.72
RUN pip install opencv_python-4.7.0.72-cp310-cp310m-manylinux2014_x86_64.whl

COPY . .
RUN mkdir -p /app/images

EXPOSE 5000

CMD ["python", "app.py"]