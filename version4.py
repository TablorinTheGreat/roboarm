import socket
from copy import deepcopy
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage

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

colorUpdateFlag=True
lower_color_extension = 0.5
upper_color_extension = 1.5

threshold = 60
previousAngle=0

MAX_SIM_THRESHOLD=4000

client=None
port= 1729
ip="0.0.0.0"

weights=[5 , 5, 2, 3, 1, 1, 1]

handOrigin=[195,250]
outline=cv2.imread("startOutline.jpg")
start=cv2.imread("start.png",cv2.IMREAD_GRAYSCALE)
start_box=[131, 96, 158, 324]
outlineNum=18806*0.9
startHand=[handOrigin,0.487654,1458,None]

camera = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

face_cascade = cv2.CascadeClassifier('haar_cascade_frontaface.xml')

bg3 = np.zeros((480,640,3), np.uint8)
bg1 = np.zeros((480,640), np.uint8)

robo=bg3.copy()
p=(int(robo.shape[1]/2))
origin=(int(robo.shape[1]/2), int(robo.shape[0]))
cv2.line(robo, origin, (p,200), (0, 0, 255), 2)
cv2.imshow("robo",robo)


titles=["x","y","aspect ratio","size","r","g","b"]
info=[[],[],[],[],[],[],[]]
info2=[]
handTitles={"loc":0,"aspect_ratio":1,"size":2,"mean_color":3}
hand=deepcopy(startHand)
last_box=start_box.copy()

handOnScene=0

(grabbed, frame) = camera.read()
cv2.rectangle(frame, (last_box[0], last_box[1]), (last_box[0]+last_box[2],last_box[1]+last_box[3]), (50, 50, 50), cv2.FILLED)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
fgbg.apply(frame)
skinbg = cv2.createBackgroundSubtractorMOG2(varThreshold=6,detectShadows=False)
stopped=0.7

out = cv2.VideoWriter('lastOutput.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (640,480))
frameCount=0

clear_background_ITERATIONS=2
clear_background_MEDIAN_BLUR=7

color_segmentation_ITERATIONS=4
color_segmentation_IMG_GAUSSIAN_BLUR=7
color_segmentation_MEDIAN_BLUR=7
color_segmentation_SKIN_MASK_GAUSSIAN_BLUR=5



def flat(lst):
    lst[0:1]=lst[0]
    lst[4:5]=lst[4]
    return lst
def addInfo(hand):
    global info
    for i in range(len(hand)):
        info[i].append(hand[i])
def roboRot(d):
    robo=bg3.copy()
    x=origin[0]+200*math.cos(math.radians(d))
    y=origin[1]-200*math.sin(math.radians(d))
    cv2.line(robo, origin, (int(x), int(y)), (0, 0, 255), 2)
    cv2.imshow("robo", robo)
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
    if handOnScene==1:
        background = Mask(frame,curr).astype(np.uint8) + \
                 (stopped * Mask(fgbg.getBackgroundImage(),invcurr)).astype(np.uint8) + \
                 ((1 - stopped) * Mask(frame,invcurr)).astype(np.uint8)
        skinBackground = Mask(skin, curr).astype(np.uint8) + \
                 (stopped * Mask(skinbg.getBackgroundImage(), invcurr)).astype(np.uint8) + \
                 ((1 - stopped) * Mask(skin, invcurr)).astype(np.uint8)
    else:
        background=frame
        skinBackground=skin
    fgbg.apply(background)
    skinbg.apply(skinBackground)
    cv2.imshow("sdcf",background)
    cv2.imshow("sdfsd",fgbg.getBackgroundImage())
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
def findHand(andmask,frame):
    global colorUpdateFlag
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
    if maxSim is not None and maxSim<MAX_SIM_THRESHOLD:
        if maxSim>=0.7*MAX_SIM_THRESHOLD:
            colorUpdateFlag=True
        info2.append(maxSim)
        print(maxSim)
        return handMask.astype(np.uint8),handBlob,True,BRec
    print(-1)
    info2.append(-100)
    return None,None,False,None
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
def clear_background(skinMask,moveMask):
    andMask=cv2.bitwise_and(skinMask,moveMask)
    andMask = cv2.medianBlur(andMask, clear_background_MEDIAN_BLUR)
    andMask = cv2.erode(andMask, kernel, iterations=clear_background_ITERATIONS)
    andMask = cv2.dilate(andMask, kernel, iterations=clear_background_ITERATIONS)
    fillMask(andMask)
    cv2.imshow("and",andMask)
    return andMask
def getHandMask(skinMask,moveMask,frame):
    global hand,last_box
    clearedBackground=clear_background(skinMask,moveMask)
    handMask, handblob, sucsses, BRec = findHand(clearedBackground, frame)
    if sucsses:
        addInfo(flat(hand.copy()))
        last_box = BRec
        hand = handblob
        return handMask
    return None
def updateColors(YCbCr_image,HSV_image,Lab_image,mask,gray):
    global lower_YCbCr_values,upper_YCbCr_values,lower_HSV_values,upper_HSV_values,lower_Lab_values,upper_Lab_values,lower_color_extension,q
    maxrange, minrange = colorRangeindexes(Mask(gray, mask))
    lower_YCbCr_values = (YCbCr_image[minrange[0]][minrange[1]] * lower_color_extension).clip(lower_YCbCr_valueso, 255)
    upper_YCbCr_values = (YCbCr_image[maxrange[0]][maxrange[1]] * upper_color_extension).clip(0, upper_YCbCr_valueso)
    lower_Lab_values = (Lab_image[minrange[0]][minrange[1]] * lower_color_extension).clip(lower_Lab_valueso, 255)
    upper_Lab_values = (Lab_image[maxrange[0]][maxrange[1]] * upper_color_extension).clip(0, upper_Lab_valueso)
    lower_HSV_values = (HSV_image[minrange[0]][minrange[1]] * lower_color_extension).clip(lower_HSV_valueso, 255)
    upper_HSV_values = (HSV_image[maxrange[0]][maxrange[1]] * upper_color_extension).clip(0, upper_HSV_values)

def color_segmentation(img,mask=None,gray=None):
    global colorUpdateFlag
    blured = cv2.GaussianBlur(img, (color_segmentation_IMG_GAUSSIAN_BLUR, color_segmentation_IMG_GAUSSIAN_BLUR), 0)
    YCbCr_image = cv2.cvtColor(blured, cv2.COLOR_BGR2YCrCb)
    HSV_image = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)
    Lab_image = cv2.cvtColor(blured, cv2.COLOR_BGR2Lab)
    if mask is not None and colorUpdateFlag:
        updateColors(YCbCr_image,HSV_image,Lab_image,mask,gray)
        colorUpdateFlag=False

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
def cropImage(image,Brec):
    return image[Brec[1]:Brec[1] + Brec[3],Brec[0] :Brec[0] + Brec[2]]
def extractAngle(handMask):
    #croped=cropImage(handMask,last_box)
    lines = cv2.HoughLines(handMask, 1, np.pi / 180, threshold=threshold)
    if lines is not None:
        ans=0
        for r, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            im = cv2.cvtColor(handMask, cv2.COLOR_GRAY2BGR)
            cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow("sdfsdf",im)
            if (x1 - x2) != 0:
                m = (y1 - y2) / (x1 - x2)
                ans=math.degrees(math.atan(m))
            else:
                ans=90
            ans=-ans
            if ans<0:
                ans=180+ans
            return ans
def sendAngle(ang):
    global previousAngle
    if ang is not None:
        if math.fabs(ang-previousAngle)>5:
            # print(ang)
            roboRot(ang)
            ang=str(ang)
            data=ang.encode('ascii')
            client.send(data)
        previousAngle=float(ang)
def setupServer():
    global client
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip,port))
    print("server started")
    server_socket.listen(1)
    print("waiting...")
    client, address = server_socket.accept()
    print("connected!")
def restart():
    global colorUpdateFlag,handOnScene,hand,last_box
    if handOnScene==1:
        colorUpdateFlag = True
        handOnScene = 0
        hand[handTitles["loc"]] = handOrigin
        roboRot(90)
        last_box = start_box.copy()
        hand = deepcopy(startHand)
roboRot(90)


handMask=start.copy()
skinbg.apply(Mask(frame,color_segmentation(frame)))
setupServer()
while True:
    (grabbed, frame) = camera.read()
    moveMask=MovementMask(frame,Mask(frame,color_segmentation(frame)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (int(0.9*x), int(0.7*y)),( int((x + w)), int(1.2*(y + h))), (255, 255, 0), cv2.FILLED)
    if handOnScene==1 and colorUpdateFlag:
        skinMask = color_segmentation(frame,handMask,gray)
    else:
        skinMask=color_segmentation(frame)
    # skin = Mask(frame,skinMask)
    if handOnScene==1:
        frameCount+=1
        cv2.putText(frame, str(frameCount), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
        handMask = getHandMask(skinMask,moveMask,frame)
        if handMask is not None:
            cv2.circle(frame, (hand[handTitles["loc"]][0], hand[handTitles["loc"]][1]), 7, (0, 255, 255), -1)
            cv2.rectangle(frame, (last_box[0], last_box[1]), (last_box[0] + last_box[2], last_box[1] + last_box[3]), (255, 100, 0), 2)
            ang=extractAngle(handMask)
            sendAngle(ang)
        else:
           restart()
    else:
        frame = cv2.add(frame, outline)
        currStartHand = Mask(Mask(frame,clear_background(skinMask,moveMask)), start)
        curr = np.count_nonzero(currStartHand)
        if curr >= outlineNum and cv2.waitKey(1) & 0xFF == ord("s"):
            hand_color = cv2.bitwise_and(frame, frame, mask=start)
            hand[handTitles["mean_color"]]=meanColor(hand_color)
            hand[handTitles["size"]]=curr
            hand[handTitles["mean_color"]]=meanColor(hand_color)
            currStartHand_gray=cv2.cvtColor(currStartHand, cv2.COLOR_BGR2GRAY)
            [py, px] = ndimage.measurements.center_of_mass(currStartHand_gray)
            hand[handTitles["loc"]]=[py,px]
            x, y, w, h = cv2.boundingRect(currStartHand_gray)
            aspect_ratio = float(w) / h
            hand[handTitles["aspect_ratio"]]=aspect_ratio
            handOnScene = 1

    out.write(frame)
    # cv2.imshow("images", np.hstack((frame, skin)))
    cv2.imshow("images", frame)

    key=cv2.waitKey(1)
    if  key& 0xFF == ord("q"):
        break
    if key & 0xFF == ord("r"):
        restart()


camera.release()
cv2.destroyAllWindows()
client.close()
for i in range(len(info)):
    plt.subplot(8, 1, i+1)
    plt.plot(info[i], label=titles[i])
    plt.legend()
plt.subplot(8, 1, 8)
plt.plot(info2, label="max sim")
plt.legend()
plt.show()
