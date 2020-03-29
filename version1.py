import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage

alpha=0
outlineNum=18806*0.85
outline=cv2.imread("startOutline.jpg")
start=cv2.imread("start.png",cv2.IMREAD_GRAYSCALE)
camera = cv2.VideoCapture(0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
face_cascade = cv2.CascadeClassifier('haar_cascade_frontaface.xml')
bg3 = np.zeros((480,640,3), np.uint8)
bg1 = np.zeros((480,640), np.uint8)
robo=bg3.copy()
p=(int(robo.shape[1]/2))
origin=(int(robo.shape[1]/2), int(robo.shape[0]))
cv2.line(robo, origin, (p,200), (0, 0, 255), 2)
#cv2.imshow("robo",robo)
roborot=300
handOrigin=[195,250]
hand={"loc":handOrigin,"aspect_ratio":0.487654,"size":1458,"mean_color":None}
s=0
s2=0
s3=0
threshold = 60
minLineLength = 10
fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg.setDetectShadows(False)
(grabbed, frame) = camera.read()
last_box=[131, 96, 158, 324]
frame2=frame.copy()
cv2.rectangle(frame2, (last_box[0], last_box[1]), (last_box[0]+last_box[2],last_box[1]+last_box[3]), (50, 50, 50), cv2.FILLED)
fgbg.apply(frame2)
stopped=0.7
def flat(lst):
    i=0
    while i<len(lst):
        while True:
            try:
                lst[i:i+1] = lst[i]
            except (TypeError, IndexError):
                break
        i += 1
    return lst
def meanColor(frame):
    return frame.sum(axis=(0, 1)) / np.count_nonzero(frame, axis=(0, 1))
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
def MovementMask(frame,mask):
    global last_box
    if s is 0:
        background=frame.copy()
        #cv2.rectangle(background, (last_box[0], last_box[1]), (last_box[0] + last_box[2], last_box[1] + last_box[3]),frame[last_box[1]:last_box[1] + last_box[3],last_box[0]:last_box[0] + last_box[2]].mean(axis=(0,1)), cv2.FILLED)
    else:
        if mask is not None:
            [x, y, w, h],han = get_mROI(mask)
            x = int((x + last_box[0]) / 2)
            y = int((y + last_box[1]) / 2)
            w = int((w + last_box[2]) / 2)
            h = int((h + last_box[3]) / 2)
            last_box = [x, y, w, h]
        curr = np.ones((frame.shape[0], frame.shape[1]), np.uint8) * 255
        cv2.rectangle(curr, (last_box[0], last_box[1]), (last_box[0] + last_box[2], last_box[1] + last_box[3]), (0, 0, 0),cv2.FILLED)
        background = cv2.bitwise_and(frame, frame, mask=curr).astype(np.uint8) +\
                     (stopped * cv2.bitwise_and(fgbg.getBackgroundImage(), fgbg.getBackgroundImage(),mask=(255 - curr))).astype(np.uint8) + \
                     ((1 - stopped) * cv2.bitwise_and(frame, frame, mask=(255 - curr))).astype(np.uint8)
    fgbg.apply(background)
    return  fgbg.apply(frame, learningRate=0)
def roboRot(d):
    global roborot
    robo=bg3.copy()
    roborot+=d
    x=origin[0]+200*math.cos(math.radians(roborot))
    y=origin[1]+200*math.sin(math.radians(roborot))
    cv2.line(robo, origin, (int(x), int(y)), (0, 0, 255), 2)
    #cv2.imshow("robo", robo)
def cords(c):
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return [cX,cY]
def cosineSim(p1,p2):
    ab=np.dot(p1,p2)
    aa=math.sqrt(np.dot(p1,p1))
    bb=math.sqrt(np.dot(p2,p2))
    return ab/(aa*bb)
def dist(p1,p2):
    return math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))
def findHand(andmask,frame,handm):
    cv2.imshow("asd",andmask)
    maxSim=None
    handBlob=None
    handMask=None
    ret, markers = cv2.connectedComponents(andmask)
    flathand=flat(list(hand.values()))
    for i in range(1, ret):
        c = markers.copy()
        c[c != i] = 0
        c[c!=0]=255
        size=np.count_nonzero(c)
        [py,px] = ndimage.measurements.center_of_mass(c)
        if size>=hand["size"]*0.7 and dist([px,py],hand["loc"])<100:
            x, y, w, h = cv2.boundingRect(c.astype(np.uint8))
            aspect_ratio = float(w) / h
            meanC=meanColor(cv2.bitwise_and(frame,frame,mask=c.astype(np.uint8)))
            blob = {"loc": [int(px),int(py)], "aspect_ratio": aspect_ratio, "size": size, "mean_color": meanC}
            flatblob=flat(list(blob.values()))
            sim=cosineSim(flatblob,flathand)
            if handBlob is None:
                maxSim = sim
                handMask = c.copy()
                handBlob=blob
            elif(sim>maxSim):
                maxSim=sim
                handMask=c.copy()
                handBlob=blob
    if maxSim is not None: #and minDist<100:
        handMask[handMask!=0]=255
        handMask=handMask.astype(np.uint8)
        return handMask,handBlob,True
    return handm,hand,False
def clear_background(skinMask,moveMask,frame,handm):
    global hand,bg1
    andMask=cv2.bitwise_and(skinMask,moveMask)
    handMask, handblob,sucsses=findHand(andMask,frame,handm)
    [x, y, w, h]=cv2.boundingRect(handMask.astype(np.uint8))
    b=bg1.copy()
    cv2.rectangle(b, (x, y), (x + w , y + h ), (255, 255,255), cv2.FILLED)
    hand=handblob
    return cv2.bitwise_and(skinMask,skinMask,mask=b)

def color_segmentation(img):
    lower_HSV_values = np.array([0, 40, 0], dtype = "uint8")
    upper_HSV_values = np.array([25, 255, 255], dtype = "uint8")

    lower_YCbCr_values = np.array((0, 138, 67), dtype = "uint8")
    upper_YCbCr_values = np.array((255, 173, 133), dtype = "uint8")

    lower_Lab_values = np.array([35, 135, 120], dtype="uint8")
    upper_Lab_values = np.array([200, 180, 155], dtype="uint8")

    blured = cv2.GaussianBlur(img, (7, 7), 0)
    YCbCr_image = cv2.cvtColor(blured, cv2.COLOR_BGR2YCrCb)
    HSV_image = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)
    Lab_image = cv2.cvtColor(blured, cv2.COLOR_BGR2Lab)

    #A binary mask is returned. White pixels (255) represent pixels that fall into the upper/lower.
    mask_YCbCr = cv2.inRange(YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
    mask_HSV = cv2.inRange(HSV_image, lower_HSV_values, upper_HSV_values)
    mask_Lab = cv2.inRange(Lab_image, lower_HSV_values, upper_HSV_values)
    skinMask=cv2.add(mask_HSV, mask_YCbCr, mask_Lab)

    skinMask = cv2.medianBlur(skinMask, 7)
    skinMask = cv2.erode(skinMask, kernel)
    skinMask = cv2.dilate(skinMask, kernel,iterations=4)
    skinMask = cv2.erode(skinMask, kernel,iterations=4)
    skinMask=cv2.GaussianBlur(skinMask, (5, 5), 0)
    sdf, contour, hier = cv2.findContours(skinMask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(skinMask, [cnt], 0, 255, -1)
    return skinMask
roboRot(0)
handMask=start.copy()
while True:
    (grabbed, frame) = camera.read()
    moveMask=MovementMask(frame,handMask)
    cv2.imshow("Azsdfghjkl",moveMask)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x-20, y-30), (x + w+20, y + h+70), (255, 0, 0), cv2.FILLED)
    skinMask=color_segmentation(frame)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)
    if s==1:
        handMask = clear_background(skinMask,moveMask,frame,handMask)
        if handMask is not None:
            cv2.circle(frame, (hand["loc"][0], hand["loc"][1]), 7, (0, 255, 255), -1)
           # cv2.imshow("images1", np.hstack((moveMask,skinMask,handMask)))
            cv2.imshow("imagdes1", handMask)
            '''lines = cv2.HoughLines(handMask, 1, np.pi / 180, threshold=threshold)
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
                    if (x1 - x2) != 0:
                        m = (y1 - y2) / (x1 - x2)
                    if m * alpha != -1:
                        ans = math.degrees(math.atan((alpha - m) / (1 + m * alpha)))
                    if s2==1 and math.fabs(ans) > 1:
                        roboRot(-ans)
                        print(ans)
                    elif s2==0:s2=1
                    alpha = m
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)'''
        else:
            s = 0
            hand["loc"] = handOrigin
            roborot =300
            roboRot(0)
    else:
        frame2 = frame.copy()
        cv2.rectangle(frame2, (last_box[0], last_box[1]), (last_box[0] + last_box[2], last_box[1] + last_box[3]),(50, 50, 50), cv2.FILLED)
        fgbg.apply(frame2)
        frame = cv2.add(frame, outline)
        startCount = cv2.bitwise_and(skinMask, skinMask, mask=start)
        curr = np.count_nonzero(startCount)
        if curr >= outlineNum:
            hand_color = cv2.bitwise_and(frame, frame, mask=start)
            hand["mean_color"]=meanColor(hand_color)
            s = 1


    cv2.rectangle(frame, (last_box[0], last_box[1]), (last_box[0] + last_box[2], last_box[1] + last_box[3]), (255, 100, 0), 2)
    cv2.imshow("images", np.hstack((frame, skin)))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()