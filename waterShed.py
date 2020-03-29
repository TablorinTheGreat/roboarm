import cv2
import numpy as np
import random as rng
rng.seed(12345)
from scipy import ndimage
kernel = np.ones((3,3),np.uint8)
c=cv2.VideoCapture(0)
lower = np.array([35, 135, 120], dtype = "uint8")
upper = np.array([150, 180, 155], dtype = "uint8")
colors = []
img=cv2.imread("water_thresh.jpg")
im=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
for co in range(30):
        colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
while True:
    ''''(g,img)=c.read()
    converted = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    skinMask = cv2.inRange(converted, lower, upper)
    skinMask = cv2.medianBlur(skinMask, 7)
    skinMask = cv2.erode(skinMask, kernel)
    skinMask = cv2.dilate(skinMask, kernel, iterations=4)
    skinMask = cv2.erode(skinMask, kernel, iterations=4)
    skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
    cv2.imshow("asdfgh",skinMask)
    sdf,contour, hier = cv2.findContours(skinMask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(skinMask, [cnt], 0, 255, -1)
    im=skinMask'''
    su=cv2.dilate(im,kernel,iterations=2)
    sure_bg = cv2.dilate(su,kernel)
    dist_transform = cv2.distanceTransform(im, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imshow("dd", dist_transform)
    dist=dist_transform
    ret, sure_fg = cv2.threshold(dist,0.7*dist.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,su)
    ret, markers = cv2.connectedComponents(sure_fg)
    seeMyLabels=cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow("Labels", seeMyLabels)
    markers[unknown == 255] = 255
    cv2.imshow("Labdels", unknown)
    markers = cv2.watershed(img,markers)
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i, j]
            if index > 0 and index <= ret:
                img[i, j, :] = colors[index - 1]
    cv2.imshow("s",img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
c.release()
cv2.destroyAllWindows()