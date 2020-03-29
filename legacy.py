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


alpha=0
outlineNum=18806*0.9

weightsDict={"loc":[5 ,5],"aspect_ratio":2,"size":2,"mean_color":[1,1,1]}

outline=cv2.imread("startOutline.jpg")
start=cv2.imread("start.png",cv2.IMREAD_GRAYSCALE)

camera = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
EdgeKernel=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],np.int32)

face_cascade = cv2.CascadeClassifier('haar_cascade_frontaface.xml')

bg3 = np.zeros((480,640,3), np.uint8)
bg1 = np.zeros((480,640), np.uint8)

IMAGE_HEIGHT=480
IMAGE_WEIGHT=640

robo=bg3.copy()
p=(int(robo.shape[1]/2))
origin=(int(robo.shape[1]/2), int(robo.shape[0]))
cv2.line(robo, origin, (p,200), (0, 0, 255), 2)
##cv2.imshow("robo",robo)
roborot=300

handOrigin=[195,250]
titleDict={1:"x",2:"y",3:"aspect ratio",4:"size",5:"r",6:"g",7:"b"}
titles={"loc":[1,2],"aspect_ratio":3,"size":4,"mean_color":[5,6,7]}
info=[[],[],[],[],[],[],[]]
info2=[]
hand={"loc":handOrigin,"aspect_ratio":0.487654,"size":1458,"mean_color":None}

s=0
s2=0
threshold = 60
minLineLength = 10

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

lower_color_extension = 0.75
upper_color_extension = 1.25
meanHSV=[]
meanLab=[]
meanYCbCr=[]
HSVspread=np.array([7,55,65])
YCbCrSpread=np.array([65,9,17])
LabSpread=np.array([42,12,9])



self.robo = self.bg3.copy()
self.p = (int(self.robo.shape[1] / 2))
self.origin = (int(self.robo.shape[1] / 2), int(self.robo.shape[0]))
cv2.line(self.robo, self.origin, (self.p, 200), (0, 0, 255), 2)
cv2.imshow("robo", self.robo)

def validh(x):
    if x<0:
        return 0
    if x>IMAGE_HEIGHT:
        return IMAGE_HEIGHT
    return x
def validw(x):
    if x < 0:
        return 0
    if x > IMAGE_WEIGHT:
        return IMAGE_WEIGHT
    return x
'''def cropImage(image):
    offset=int(hand["size"]*0.05)
    return image[validh(last_box[1]-offset):validh(last_box[1]+last_box[3]+offset),validw(last_box[0]-offset):validw(last_box[0]+last_box[2]+offset)
'''
def edegeDetection(gray):
    edges = cv2.Canny(gray, 10, 70)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    edges = cv2.dilate(edges,kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=2)
    '''dist_transform = cv2.distanceTransform(edges, cv2.DIST_L2, 5)
cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
cv2.imshow("dd", dist_transform)
edge= cv2.filter2D(gray,-1, EdgeKernel)
 edge = cv2.dilate(edge, None)
    edge = cv2.erode(edge, None)
    #edge = cv2.GaussianBlur(edge, (5, 5), 0)'''
    #ret3, edge = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges
def cropImage(image,Brec):
    return image[Brec[1]:Brec[1] + Brec[3],Brec[0] :Brec[0] + Brec[2]]
def dist(self,p1,p2):
        return math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))
def cosineSim(p1,p2):
    ab=np.dot(p1,p2)
    aa=math.sqrt(np.dot(p1,p1))
    bb=math.sqrt(np.dot(p2,p2))
    return ab/(aa*bb)

def norm(p):
    return math.sqrt(np.dot(p,p))
def blobSeparation(gray,frame,UnSepHandMask,Brec):
    maxSim = None
    handBlob = None
    BRec = None
    handMask = None
    #frame=cropImage(frame,Brec)
    edges =edegeDetection(gray)
    UnSepHandMaskDial = cv2.dilate(UnSepHandMask, kernel)
    edges=cv2.bitwise_and(edges,UnSepHandMaskDial)
    sdf, contours, hier = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame,contours, 0, (50,200,0), 2)
    cv2.imshow("sdfg",edges)
    flathand = flat(list(hand.values()))
    '''for c in contours:
        size=c.size
        [py,px] = cords(c)
        if py is not None:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            bg=bg1.copy()
            cv2.drawContours(bg, [c], 0, 255, -1)
            mask=Mask(frame,bg)
            meanC=meanColor(mask)
            blob = {"loc": [int(px),int(py)], "aspect_ratio": aspect_ratio, "size": size, "mean_color": meanC}
            flatblob=flat(list(blob.values()))
            sim=simi(np.array(flatblob),np.array(flathand))
            if handBlob is None:
                maxSim = sim
                handBlob=blob
                BRec=[x,y,w,h]
                handMask=mask
            elif(sim<maxSim):
                maxSim=sim
                handBlob=blob
                BRec = [x, y, w, h]
                handMask = mask'''
    if maxSim is not None:#figure out what's causing the none returnq
        return handMask,handBlob,True,BRec
    return UnSepHandMask,hand,False,last_box
def lowerColor(frame):
    channel1=frame[:,:,0]
    channel2=frame[:,:,1]
    channel3=frame[:,:,2]
    channel1=channel1[np.nonzero(channel1)]
    channel2=channel2[np.nonzero(channel2)]
    channel3=channel3[np.nonzero(channel3)]
    channel1=np.amin(channel1)
    channel2=np.amin(channel2)
    channel3=np.amin(channel3)
    return np.array([channel1,channel2,channel3],dtype="uint8")
def upperColor(frame):
    channel1=frame[:,:,0]
    channel2=frame[:,:,1]
    channel3=frame[:,:,2]
    channel1=channel1[np.nonzero(channel1)]
    channel2=channel2[np.nonzero(channel2)]
    channel3=channel3[np.nonzero(channel3)]
    channel1=np.amax(channel1)
    channel2=np.amax(channel2)
    channel3=np.amax(channel3)
    return np.array([channel1,channel2,channel3],dtype="uint8")
def medianColor(frame):
    channel1=frame[:,:,0]
    channel2=frame[:,:,1]
    channel3=frame[:,:,2]
    channel1=channel1[np.nonzero(channel1)]
    channel2=channel2[np.nonzero(channel2)]
    channel3=channel3[np.nonzero(channel3)]
    channel1=np.median(channel1)
    channel2=np.median(channel2)
    channel3=np.median(channel3)
    return np.array([channel1,channel2,channel3])



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
def updateColorValues(median,spread):
    return (median+spread).clip(0,255)
 def roboRot(self,d):
        robo= self.bg3.copy()
        x= self.origin[0]+200*math.cos(math.radians(d))
        y= self.origin[1]-200*math.sin(math.radians(d))
        cv2.line(robo,  self.origin, (int(x), int(y)), (0, 0, 255), 2)
        cv2.imshow("robo", robo)
def cords(self,c):
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return [cX,cY]
        return [None,None]