import math

import cv2
import numpy as np
num=18806*0.85
im=cv2.imread("startOutline.jpg")
start=cv2.imread("start.png",cv2.IMREAD_GRAYSCALE)
camera = cv2.VideoCapture(0)
print(np.count_nonzero(start))
lower = np.array([35, 135, 120], dtype = "uint8")
upper = np.array([200, 180, 155], dtype = "uint8")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
face_cascade = cv2.CascadeClassifier('haar_cascade_frontaface.xml')
hand=[195,251]
s=0
def cords(c):
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return [cX,cY]
def dist(p1,p2):
    return math.sqrt(math.pow((p1[0]-p2[0]),2)+math.pow((p1[1]-p2[1]),2))
def findHand(con):
    minDist=None
    handBlob=None
    for c in con:
        cor=cords(c)
        if cor is not None:
            d=dist(cor, hand)
            if handBlob is None:
                minDist = d
                handBlob = c
            elif(d<minDist):
                minDist=d
                handBlob=c
    if minDist<50:
        return handBlob
def clear_background(skinMask):
    global hand
    im2, contours, hierarchy = cv2.findContours(skinMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    h=findHand(contours)
    hand=cords(h)
    if hand is not None:
        bg=np.zeros(skinMask.shape,np.uint8)
        cv2.drawContours(bg, [h], -1, 255, -1)
        return bg
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
    if s==0:
        y=cv2.bitwise_and(skinMask, skinMask, mask=start)
        cv2.imshow("fd",y)
        curr=np.count_nonzero(y)
        if curr>=num:
            s=1
    else:
        clear_background(skinMask)
        cv2.circle(frame, (hand[0], hand[1]), 7, (0, 255, 255), -1)
    o=cv2.add(frame,im)
    cv2.imshow("d",o)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()