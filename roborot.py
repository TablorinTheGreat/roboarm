import math
from threading import Thread
import numpy as np
import cv2
from time import sleep
roborot=300
im=cv2.imread("start.png",cv2.IMREAD_GRAYSCALE)
rows,cols = im.shape
meanx=meany=meanxy=meanxx=0
for i in range(rows):
    for j in range(cols):
        if im[i][j]==255:
            meanx+=j
            meany+=i
            meanxx+=j*j
            meanxy+=i*j
num=np.count_nonzero(im)
meanx/=num
meany/=num
meanxy/=num
meanxx/=num
m=(meanx*meany-meanxy)/(meanx*meanx-meanxx)
b=meany-m*meanx
p1=(0,int(b))
p2=(cols-1,int((cols-1)*m+b))
lines = cv2.HoughLines(im, 1, np.pi / 180, threshold=10)
im=cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
bg3 = np.zeros((480,640,3), np.uint8)
robo=bg3.copy()
p=(int(robo.shape[1]/2))
origin=(int(robo.shape[1]/2), int(robo.shape[0]))
cv2.line(robo, origin, (p,200), (0, 0, 255), 2)
cv2.imshow("robo",robo)
if lines is not None:
    for r, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

def roboRot(d):
        d=-d
        robo = bg3.copy()
        x = origin[0] + 200 * math.cos(math.radians(d))
        y = origin[1] + 200 * math.sin(math.radians(d))
        cv2.line(robo, origin, (int(x), int(y)), (0, 0, 255), 2)
        cv2.imshow("robo", robo)
i=0
while True:
    roboRot(i)
    print(i)
    i+=5
    sleep(0.1)

    if cv2.waitKey(1)&0xFF==ord("q"):
        break
cv2.destroyAllWindows()