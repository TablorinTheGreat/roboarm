import cv2
import numpy as np
import math
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
alpha=0.9
last_box=[131, 96, 158, 324]
camera = cv2.VideoCapture(0)
(grabbed,frame)=camera.read()
bgmodel=np.ones((frame.shape[0],frame.shape[1]),np.uint8)*255
cv2.rectangle(bgmodel, (last_box[0], last_box[1]), (last_box[0]+last_box[2],last_box[1]+last_box[3]), (50, 50, 50), cv2.FILLED)
bgmodel=cv2.bitwise_and(frame,frame,mask=bgmodel)
T=np.ones((frame.shape[0],frame.shape[1]),np.uint8)*30
def compBGMODEL(frame):
    global bgmodel
    curr = np.ones((frame.shape[0], frame.shape[1]), np.uint8) * 255
    cv2.rectangle(curr, (last_box[0], last_box[1]), (last_box[0] + last_box[2], last_box[1] + last_box[3]),(0, 0, 0), cv2.FILLED)
    bgmodel=(alpha*cv2.bitwise_and(bgmodel, bgmodel, mask=curr)).astype(np.uint8)+((1-alpha)*cv2.bitwise_and(frame, frame, mask=curr)).astype(np.uint8)+(cv2.bitwise_and(bgmodel, bgmodel, mask=(255-curr))).astype(np.uint8)
def movementMask(frame):
    global T
    diff = np.linalg.norm(frame.astype(np.int)-bgmodel.astype(np.int),axis=2)
    mask=diff
    mask=mask-T
    mask[mask<0]=0
    mask[mask>0]=255
    mask=mask.astype(np.uint8)
    curr = np.ones((frame.shape[0], frame.shape[1]), np.uint8) * 255
    cv2.rectangle(curr, (last_box[0], last_box[1]), (last_box[0] + last_box[2], last_box[1] + last_box[3]), (0, 0, 0),cv2.FILLED)
    T= (alpha*cv2.bitwise_and(T, T, mask=curr)).astype(np.uint8)+(5*(1-alpha)*cv2.bitwise_and(diff, diff, mask=curr)).astype(np.uint8) +(cv2.bitwise_and(T, T, mask=(255 - curr))).astype(np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    return mask
def get_mROI(mask):
    df, contour, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(contour)==0:
        return last_box
    maxc=contour[0]
    max=contour[0].size
    for i in range(1,len(contour)):
        if contour[i].size>max:
            max=contour[i].size
            maxc=contour[i]
    return cv2.boundingRect(maxc)

while True:
    (grabbed, frame) = camera.read()
    mask=movementMask(frame)
    x,y,w,h=get_mROI(mask)
    x = int((x + last_box[0]) / 2)
    y = int((y + last_box[1]) / 2)
    w = int((w + last_box[2]) / 2)
    h = int((h + last_box[3]) / 2)
    last_box=[x,y,w,h]
    compBGMODEL(frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
    cv2.imshow("bg", np.hstack((frame,bgmodel)))
    cv2.imshow("sd",np.hstack((mask,T)))
    if cv2.waitKey(1)&0xFF==ord("q"):
        break
camera.release()
cv2.destroyAllWindows()