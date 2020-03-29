import cv2
import numpy as np
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg.setDetectShadows(False)
camera=cv2.cv2.VideoCapture(0)
(grabbed, frame) = camera.read()
last_box=[131, 96, 158, 324]
frame2=frame.copy()
cv2.rectangle(frame2, (last_box[0], last_box[1]), (last_box[0]+last_box[2],last_box[1]+last_box[3]), (50, 50, 50), cv2.FILLED)
fgbg.apply(frame2)
mask = fgbg.apply(frame, learningRate=0)
alpha=0.7
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
    return cv2.boundingRect(maxc),maxc
while True:
    (grabbed, frame) = camera.read()
    [x, y, w, h],m = get_mROI(mask)
    x = int((x + last_box[0]) / 2)
    y = int((y + last_box[1]) / 2)
    w = int((w + last_box[2]) / 2)
    h = int((h + last_box[3]) / 2)
    last_box=[x,y,w,h]
    curr = np.ones((frame.shape[0], frame.shape[1]), np.uint8) * 255
    cv2.rectangle(curr, (last_box[0], last_box[1]), (last_box[0] + last_box[2], last_box[1] + last_box[3]), (0, 0, 0),cv2.FILLED)
    frame2=cv2.bitwise_and(frame, frame, mask=curr).astype(np.uint8)+(alpha*cv2.bitwise_and(fgbg.getBackgroundImage(), fgbg.getBackgroundImage(), mask=(255-curr))).astype(np.uint8)+((1-alpha)*cv2.bitwise_and(frame, frame, mask=(255-curr))).astype(np.uint8)
    cv2.imshow("ifmages", frame2)
    fgbg.apply(frame2)
    mask=fgbg.apply(frame,learningRate=0)
    cv2.drawContours(frame, m, -1, 255, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
    cv2.imshow("images", mask)
    cv2.imshow("imagerts",np.hstack((frame,fgbg.getBackgroundImage())))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()