import cv2
import numpy as np
outlineNum=18806*0.85
im=cv2.imread("startOutline.jpg")
start=cv2.imread("start.png",cv2.IMREAD_GRAYSCALE)
camera = cv2.VideoCapture(0)
print(np.count_nonzero(start))
lower = np.array([35, 135, 120], dtype = "uint8")
upper = np.array([200, 180, 155], dtype = "uint8")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
face_cascade = cv2.CascadeClassifier('haar_cascade_frontaface.xml')
while True:
    (grabbed, frame) = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x - 20, y - 30), (x + w + 20, y + h + 70), (255, 0, 0), cv2.FILLED)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    skinMask = cv2.inRange(converted, lower, upper)
    skinMask = cv2.medianBlur(skinMask, 7)
    skinMask = cv2.erode(skinMask, kernel)
    skinMask = cv2.dilate(skinMask, kernel, iterations=4)
    skinMask = cv2.erode(skinMask, kernel, iterations=4)
    cv2.imshow("f",skinMask)
    y=cv2.bitwise_and(skinMask, skinMask, mask=start)
    cv2.imshow("fd",y)
    curr=np.count_nonzero(y)
    if curr>=outlineNum:
        print("start")
    o=cv2.add(frame,im)
    cv2.imshow("d",o)
    if cv2.waitKey(1) & 0xFF == ord("q") or curr>=outlineNum:
        break
camera.release()
cv2.destroyAllWindows()