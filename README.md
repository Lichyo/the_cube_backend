# CyCube-Server
## Installation // original
    pip isntall -r requirements.txt
    python app.py

## Installation with Docker
Build Image

    docker build --tag cy-cube .
Run Container

    docker run -p 5000:5000 cy-cube