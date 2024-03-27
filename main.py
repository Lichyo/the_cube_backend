import cv2
import color_detection as cd

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break

    img = cv2.resize(img, (640, 360))
    img = cd.process_image(img=img, color=cd.orange)
    img = cd.process_image(img=img, color=cd.red)
    img = cd.process_image(img=img, color=cd.yellow)
    img = cd.process_image(img=img, color=cd.white)
    img = cd.process_image(img=img, color=cd.blue)
    img = cd.process_image(img=img, color=cd.green)

    cv2.imshow('cube', img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
