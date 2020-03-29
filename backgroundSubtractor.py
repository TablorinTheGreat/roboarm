import cv2
import numpy as np
import math
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
alpha=0.9
camera = cv2.VideoCapture(0)
(grabbed,frame)=camera.read()
bgmodel=[frame]
history=[cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.int)]
(grabbed,frame)=camera.read()
history.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.int))
model=frame
last_box=[]
def compBGMODEL(bg):
    model=(alpha*bg[0]).astype(np.uint8)+((1-alpha)*bg[1]).astype(np.uint8)
    for i in range(2,len(bg)-1):
        model=(alpha*model).astype(np.uint8)+((1-alpha)*bg[i]).astype(np.uint8)
    mask=movementMask(frame)
    invmask=255-mask
    cframe=cv2.bitwise_and(bg[len(bg)-1],bg[len(bg)-1],mask=invmask)
    static=cv2.bitwise_and(model,model,mask=mask)
    curr=cframe+static
    cv2.imshow("sdf",curr)
    model = (alpha * model).astype(np.uint8) + ((1 - alpha) * curr).astype(np.uint8)
    return model
def movementMask(frame):
    diff=frame-model
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff[diff>=128]=255
    diff[diff<128]=0
    diff=cv2.medianBlur(diff,7)
    #th = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return diff
def get_mROI(frame):
    diff=[]
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.int)
    for i in history:
        d=i-frame
        np.abs(d)
        d[d<8]=0
        d[d>=8]=255
        diff.append(d)
    history.remove(history[0])
    history.append(frame)
    mask=cv2.bitwise_and(diff[0],diff[1]).astype(np.uint8)
    mask=cv2.medianBlur(mask, 3)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    cv2.imshow("sd",mask)
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
    print(len(bgmodel))
    (grabbed,frame)=camera.read()
    '''bgmodel.append(frame)
    model=compBGMODEL(bgmodel)
    if len(bgmodel)>history:
        bgmodel.remove(bgmodel[0])'''
    x,y,w,h=get_mROI(frame)
    if len(last_box)==0:
        last_box=[x,y,w,h]
    else:
        x=int((x+last_box[0])/2)
        y=int((y+last_box[1])/2)
        w=int((w+last_box[2])/2)
        h=int((h+last_box[2])/2)
        last_box = [x, y, w, h]
    cv2.rectangle(frame, (x, y), (x + w , y + h), (255, 100, 0), 2)
    cv2.imshow("bg",frame)
    if cv2.waitKey(1)&0xFF==ord("q"):
        break
camera.release()
cv2.destroyAllWindows()