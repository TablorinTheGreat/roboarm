import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage
from queue import Queue
from threading import Thread
import time
lower_HSV_values = np.array([0, 40, 0], dtype = "uint8")
upper_HSV_values = np.array([25, 255, 255], dtype = "uint8")

lower_YCbCr_values = np.array((0, 138, 67), dtype = "uint8")
upper_YCbCr_values = np.array((255, 173, 133), dtype = "uint8")

lower_Lab_values = np.array([35, 135, 120], dtype="uint8")
upper_Lab_values = np.array([200, 180, 155], dtype="uint8")

lower_HSV_valueso = np.array([0, 40, 0], dtype = "uint8")
upper_HSV_valueso= np.array([25, 255, 255], dtype = "uint8")

lower_YCbCr_valueso = np.array((0, 138, 67), dtype = "uint8")
upper_YCbCr_valueso = np.array((255, 173, 133), dtype = "uint8")

lower_Lab_valueso = np.array([35, 135, 120], dtype="uint8")
upper_Lab_valueso = np.array([200, 180, 155], dtype="uint8")


angleDelta=0
outlineNum=18806*0.9

weights=[5 , 5, 2, 3, 1, 1, 1]

outline=cv2.imread("startOutline.jpg")
start=cv2.imread("start.png",cv2.IMREAD_GRAYSCALE)

camera = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

face_cascade = cv2.CascadeClassifier('haar_cascade_frontaface.xml')

bg3 = np.zeros((480,640,3), np.uint8)
bg1 = np.zeros((480,640), np.uint8)

robo=bg3.copy()
p=(int(robo.shape[1]/2))
origin=(int(robo.shape[1]/2), int(robo.shape[0]))
cv2.line(robo, origin, (p,200), (0, 0, 255), 2)
##cv2.imshow("robo",robo)
roborot=300

handOrigin=[195,250]
titles=["x","y","aspect ratio","size","r","g","b"]
info=[[],[],[],[],[],[],[]]
info2=[]
handTitles={"loc":0,"aspect_ratio":1,"size":2,"mean_color":3}
hand=[handOrigin,0.487654,1458,None]

handOnScene=0

last_box=[131, 96, 158, 324]

(grabbed, frame) = camera.read()
cv2.rectangle(frame, (last_box[0], last_box[1]), (last_box[0]+last_box[2],last_box[1]+last_box[3]), (50, 50, 50), cv2.FILLED)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
fgbg.apply(frame)
skinbg = cv2.createBackgroundSubtractorMOG2(varThreshold=6,detectShadows=False)

out =cv2.VideoWriter('lastOutput.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (640,480))

frameCount=0

colorUpdateInterval=15

stopped=0.7

clear_background_ITERATIONS=2
clear_background_MEDIAN_BLUR=7

color_segmentation_ITERATIONS=4
color_segmentation_IMG_GAUSSIAN_BLUR=7
color_segmentation_MEDIAN_BLUR=7
color_segmentation_SKIN_MASK_GAUSSIAN_BLUR=5

lower_color_extension = 0.7
upper_color_extension = 1.3

threadArgs=Queue()
def flat(lst):
    lst[0:1]=lst[0]
    lst[4:5]=lst[4]
    return lst
def add(hand):
    global info
    for i in range(len(hand)):
        info[i].append(hand[i])
def roboRot(d):
    global roborot
    robo=bg3.copy()
    roborot+=d
    x=origin[0]+200*math.cos(math.radians(roborot))
    y=origin[1]+200*math.sin(math.radians(roborot))
    cv2.line(robo, origin, (int(x), int(y)), (0, 0, 255), 2)
    ##cv2.imshow("robo", robo)
def fillMask(mask):
    sdf, contour, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contour, 0, 255, -1)
def Mask(img,msk):
    return cv2.bitwise_and(img,img,mask=msk)
def meanColor(frame):
    return frame.sum(axis=(0, 1)) / np.count_nonzero(frame, axis=(0, 1))
def MovementMask(frame,skin):
    global last_box,fgbg,skinbg
    invcurr= bg1.copy()
    cv2.rectangle(invcurr, (last_box[0], last_box[1]), (last_box[0] + last_box[2], last_box[1] + last_box[3]), (255, 255, 255),cv2.FILLED)
    curr=(255 - invcurr)
    background = Mask(frame,curr).astype(np.uint8) + \
                 (stopped * Mask(fgbg.getBackgroundImage(),invcurr)).astype(np.uint8) + \
                 ((1 - stopped) * Mask(frame,invcurr)).astype(np.uint8)
    skinBackground = Mask(skin, curr).astype(np.uint8) + \
                 (stopped * Mask(skinbg.getBackgroundImage(), invcurr)).astype(np.uint8) + \
                 ((1 - stopped) * Mask(skin, invcurr)).astype(np.uint8)
    fgbg.apply(background)
    skinbg.apply(skinBackground)
    return cv2.bitwise_or(fgbg.apply(frame, learningRate=0),skinbg.apply(skin, learningRate=0))
def cords(c):
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return [cX,cY]
    return [None,None]
def simi(p1,hantVec):
    return np.sum(np.multiply(((np.abs(p1-hantVec)/hantVec)*100),weights))
def dist(p1,p2):
    return math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))
def findHand(andmask,frame,handm):
    maxSim=None
    handBlob=None
    handMask=None
    BRec=None
    ret, markers = cv2.connectedComponents(andmask)
    flathand=flat(hand.copy())
    for i in range(1, ret):
        c = markers.copy()
        c[c != i] = 0
        c[c!=0]=255
        size=np.count_nonzero(c)
        [py,px] = ndimage.measurements.center_of_mass(c)
        x, y, w, h = cv2.boundingRect(c.astype(np.uint8))
        aspect_ratio = float(w) / h
        meanC=meanColor(Mask(frame,c.astype(np.uint8)))
        blob = [[int(px),int(py)], aspect_ratio, size, meanC]
        flatblob=flat(blob.copy())
        sim=simi(np.array(flatblob),np.array(flathand))
        if handBlob is None:
            maxSim = sim
            handMask = c
            handBlob=blob
            BRec=[x,y,w,h]
        elif(sim<maxSim):
            maxSim=sim
            handMask=c
            handBlob=blob
            BRec = [x, y, w, h]
    if maxSim is not None:
        info2.append(maxSim)
        return handMask.astype(np.uint8),handBlob,True,BRec
    print(-1)
    info2.append(-100)
    return handm,hand,False,last_box
def colorRangeindexes(gray):
    max=None
    min=None
    imin=None
    jmin=None
    imax=None
    jmax=None
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if max==None:
                if gray[i][j]>0:
                    max=min=gray[i][j]
                    imin=imax=i
                    jmin=jmax=j
            else:
                if gray[i][j]>max:
                    max=gray[i][j]
                    imax=i
                    jmax=j
                if gray[i][j]<min and gray[i][j]>0:
                    min=gray[i][j]
                    imin=i
                    jmin=j
    return [imax,jmax],[imin,jmin]
def clear_background(skinMask,moveMask,frame,gray,handm):
    global hand,bg1,last_box
    andMask=cv2.bitwise_and(skinMask,moveMask)
    andMask = cv2.medianBlur(andMask, clear_background_MEDIAN_BLUR)
    andMask = cv2.erode(andMask, kernel, iterations=clear_background_ITERATIONS)
    andMask = cv2.dilate(andMask, kernel, iterations=clear_background_ITERATIONS)
    fillMask(andMask)
    handMask, handblob,sucsses,BRec=findHand(andMask,frame,handm)
    add(flat(hand.copy()))
    last_box=BRec
    hand=handblob
    return handMask
def updateColorValues(median,spread):
    return (median+spread).clip(0,255)
def updateColors(YCbCr_image,HSV_image,Lab_image,mask,gray):
    global lower_YCbCr_values,upper_YCbCr_values,lower_HSV_values,upper_HSV_values,lower_Lab_values,upper_Lab_values,lower_color_extension,upper_color_extension
    maxrange, minrange = colorRangeindexes(Mask(gray, mask))
    lower_YCbCr_values = (YCbCr_image[minrange[0]][minrange[1]] * lower_color_extension).clip(lower_YCbCr_valueso, 255)
    upper_YCbCr_values = (YCbCr_image[maxrange[0]][maxrange[1]] * upper_color_extension).clip(0, upper_YCbCr_valueso)
    lower_Lab_values = (Lab_image[minrange[0]][minrange[1]] * lower_color_extension).clip(lower_Lab_valueso, 255)
    upper_Lab_values = (Lab_image[maxrange[0]][maxrange[1]] * upper_color_extension).clip(0, upper_Lab_values)
    lower_HSV_values = (HSV_image[minrange[0]][minrange[1]] * lower_color_extension).clip(lower_HSV_values, 255)
    upper_HSV_values = (HSV_image[maxrange[0]][maxrange[1]] * upper_color_extension).clip(0, upper_HSV_values)
def colorUpdateThread(args):
    global colorUpdateInterval,threadArgs
    while True:
        run=args.get()
        if run==False:
            break
        YCbCr_image=args.get()
        HSV_image=args.get()
        Lab_image=args.get()
        mask=args.get()
        gray=args.get()
        colorUpdateInterval = 50
        updateColors(YCbCr_image,HSV_image,Lab_image,mask,gray)


def color_segmentation(img,mask=None,gray=None):
    global frameCount,colorUpdateInterval,threadArgs
    blured = cv2.GaussianBlur(img, (color_segmentation_IMG_GAUSSIAN_BLUR, color_segmentation_IMG_GAUSSIAN_BLUR), 0)
    YCbCr_image = cv2.cvtColor(blured, cv2.COLOR_BGR2YCrCb)
    HSV_image = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)
    Lab_image = cv2.cvtColor(blured, cv2.COLOR_BGR2Lab)
    if mask is not None and frameCount % colorUpdateInterval == colorUpdateInterval - 1:
        threadArgs.put(True)
        threadArgs.put(YCbCr_image)
        threadArgs.put(HSV_image)
        threadArgs.put(Lab_image)
        threadArgs.put(mask)
        threadArgs.put(gray)

    mask_YCbCr = cv2.inRange(YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
    mask_HSV = cv2.inRange(HSV_image, lower_HSV_values, upper_HSV_values)
    mask_Lab = cv2.inRange(Lab_image, lower_Lab_values, upper_Lab_values)
    skinMask=cv2.add(mask_HSV, mask_YCbCr, mask_Lab)

    skinMask = cv2.medianBlur(skinMask, color_segmentation_MEDIAN_BLUR)
    skinMask = cv2.erode(skinMask, kernel)
    skinMask = cv2.dilate(skinMask, kernel,iterations=color_segmentation_ITERATIONS)
    skinMask = cv2.erode(skinMask, kernel,iterations=color_segmentation_ITERATIONS)
    skinMask=cv2.GaussianBlur(skinMask, (color_segmentation_SKIN_MASK_GAUSSIAN_BLUR, color_segmentation_SKIN_MASK_GAUSSIAN_BLUR), 0)
    fillMask(skinMask)
    return skinMask

def extractAngle():
    #use alpha
    return 0
def sendAngle():
    #send to hand
    return 0
roboRot(0)
handMask=start.copy()
skinbg.apply(Mask(frame,color_segmentation(frame)))
colorUpdateThread=Thread(target=colorUpdateThread,args=(threadArgs,))
colorUpdateThread.start()

while True:
    (grabbed, frame) = camera.read()
    moveMask=MovementMask(frame,Mask(frame,color_segmentation(frame)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (int(0.9*x), int(0.7*y)),( int((x + w)), int(1.2*(y + h))), (255, 255, 0), cv2.FILLED)
    if handOnScene==1:
        skinMask = color_segmentation(frame,handMask,gray)
    else:
        skinMask=color_segmentation(frame)
    skin = Mask(frame,skinMask)
    cv2.imshow("imagdes1", skin)
    if handOnScene==1:
        handMask = clear_background(skinMask,moveMask,frame,gray,handMask)
        frameCount+=1
        cv2.putText(frame, str(frameCount), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
        if handMask is not None:
            cv2.circle(frame, (hand[handTitles["loc"]][0], hand[handTitles["loc"]][1]), 7, (0, 255, 255), -1)
            cv2.imshow("imagdes1", handMask)
            extractAngle()
            sendAngle()
        else:
            if colorUpdateThread.is_alive():
                threadArgs.put(False)
                colorUpdateInterval=15
            handOnScene = 0
            hand[handTitles["loc"]] = handOrigin
            roborot =300
            roboRot(0)
    else:
        frame = cv2.add(frame, outline)
        startCount = Mask(skin, start)
        curr = np.count_nonzero(startCount)
        if curr >= outlineNum :
            hand_color = cv2.bitwise_and(frame, frame, mask=start)
            hand[handTitles["mean_color"]]=meanColor(hand_color)
            handOnScene = 1

    cv2.rectangle(frame, (last_box[0], last_box[1]), (last_box[0] + last_box[2], last_box[1] + last_box[3]), (255, 100, 0), 2)
    out.write(frame)
    cv2.imshow("images", np.hstack((frame, skin)))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
for i in range(len(info)):
    plt.subplot(8, 1, i+1)
    plt.plot(info[i], label=titles[i])
    plt.legend()
plt.subplot(8, 1, 8)
plt.plot(info2, label="max sim")
plt.legend()
plt.show()