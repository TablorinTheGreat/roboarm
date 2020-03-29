import socket
import threading
from copy import deepcopy
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage

client = None
port = 1729
ip = "0.0.0.0"
class camera_angle:

    def __init__(self,cameraNum,client):
        self.lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
        self.upper_HSV_values = np.array([25, 255, 255], dtype="uint8")

        self.lower_YCbCr_values = np.array((0, 138, 67), dtype="uint8")
        self.upper_YCbCr_values = np.array((255, 173, 133), dtype="uint8")

        self.lower_Lab_values = np.array([35, 135, 120], dtype="uint8")
        self.upper_Lab_values = np.array([200, 180, 155], dtype="uint8")

        self.lower_HSV_valueso = np.array([0, 40, 0], dtype="uint8")
        self.upper_HSV_valueso = np.array([25, 255, 255], dtype="uint8")

        self.lower_YCbCr_valueso = np.array((0, 138, 67), dtype="uint8")
        self.upper_YCbCr_valueso = np.array((255, 173, 133), dtype="uint8")

        self.lower_Lab_valueso = np.array([35, 135, 120], dtype="uint8")
        self.upper_Lab_valueso = np.array([200, 180, 155], dtype="uint8")

        self.colorUpdateFlag = True
        self.lower_color_extension = 0.5
        self.upper_color_extension = 1.5

        self.threshold = 60
        self.previousAngle = 0

        self.MAX_SIM_THRESHOLD = 4000

        self.weights = [5, 5, 2, 3, 1, 1, 1]

        self.handOrigin = [195, 250]
        self.outline = cv2.imread("startOutline.jpg")
        self.start = cv2.imread("start.png", cv2.IMREAD_GRAYSCALE)
        self.start_box = [131, 96, 158, 324]
        self.outlineNum = 18806 * 0.9
        self.startHand = [self.handOrigin, 0.487654, 1458, None]

        self.cameraNum=cameraNum
        self.camera = cv2.VideoCapture(self.cameraNum)

        self.client=client

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        self.face_cascade = cv2.CascadeClassifier('haar_cascade_frontaface.xml')

        self.bg3 = np.zeros((480, 640, 3), np.uint8)
        self.bg1 = np.zeros((480, 640), np.uint8)

        self.robo = self.bg3.copy()
        self.p = (int(self.robo.shape[1] / 2))
        self.origin = (int(self.robo.shape[1] / 2), int(self.robo.shape[0]))
        cv2.line(self.robo, self.origin, (self.p, 200), (0, 0, 255), 2)
        cv2.imshow("robo", self.robo)

        self.titles = ["x", "y", "aspect ratio", "size", "r", "g", "b"]
        self.info = [[], [], [], [], [], [], []]
        self.info2 = []
        self.handTitles = {"loc": 0, "aspect_ratio": 1, "size": 2, "mean_color": 3}
        self.hand = deepcopy(self.startHand)
        self.last_box = self.start_box.copy()

        self.handOnScene = 0

        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.skinbg = cv2.createBackgroundSubtractorMOG2(varThreshold=6, detectShadows=False)
        self.stopped = 0.7

        self.out = cv2.VideoWriter('lastOutput.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (640, 480))
        self.frameCount = 0

        self.clear_background_ITERATIONS = 2
        self.clear_background_MEDIAN_BLUR = 7

        self.color_segmentation_ITERATIONS = 4
        self.color_segmentation_IMG_GAUSSIAN_BLUR = 7
        self.color_segmentation_MEDIAN_BLUR = 7
        self.color_segmentation_SKIN_MASK_GAUSSIAN_BLUR = 5
    def flat(self,lst):
        lst[0:1]=lst[0]
        lst[4:5]=lst[4]
        return lst
    def addInfo(self,hand):
        for i in range(len(hand)):
            self.info[i].append(hand[i])
    def roboRot(self,d):
        robo= self.bg3.copy()
        x= self.origin[0]+200*math.cos(math.radians(d))
        y= self.origin[1]-200*math.sin(math.radians(d))
        cv2.line(robo,  self.origin, (int(x), int(y)), (0, 0, 255), 2)
        cv2.imshow("robo", robo)
    def fillMask(self,mask):
        sdf, contour, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contour, 0, 255, -1)
    def Mask(self,img,msk):
        return cv2.bitwise_and(img,img,mask=msk)
    def meanColor(self,frame):
        return frame.sum(axis=(0, 1)) / np.count_nonzero(frame, axis=(0, 1))
    def MovementMask(self,frame,skin):
        invcurr=  self.bg1.copy()
        cv2.rectangle(invcurr, (self.last_box[0], self.last_box[1]), (self.last_box[0] + self.last_box[2], self.last_box[1] + self.last_box[3]), (255, 255, 255),cv2.FILLED)
        curr=(255 - invcurr)
        if self.handOnScene==1:
            background =   self.Mask(frame,curr).astype(np.uint8) + \
                     ( self.stopped *  self.Mask( self.fgbg.getBackgroundImage(),invcurr)).astype(np.uint8) + \
                     ((1 -  self.stopped) *  self.Mask(frame,invcurr)).astype(np.uint8)
            skinBackground =  self.Mask(skin, curr).astype(np.uint8) + \
                     ( self.stopped *  self.Mask( self.skinbg.getBackgroundImage(), invcurr)).astype(np.uint8) + \
                     ((1 -  self.stopped) *  self.Mask(skin, invcurr)).astype(np.uint8)
        else:
            background=frame
            skinBackground=skin
        self.fgbg.apply(background)
        self.skinbg.apply(skinBackground)

        return cv2.bitwise_or(self.fgbg.apply(frame, learningRate=0),self.skinbg.apply(skin, learningRate=0))
    def cords(self,c):
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return [cX,cY]
        return [None,None]
    def simi(self,p1,handVec):
        return np.sum(np.multiply(((np.abs(p1-handVec)/handVec)*100),self.weights))
    def dist(self,p1,p2):
        return math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))
    def findHand(self,andmask,frame):
        maxSim=None
        handBlob=None
        handMask=None
        BRec=None
        ret, markers = cv2.connectedComponents(andmask)
        flathand=self.flat(self.hand.copy())
        for i in range(1, ret):
            c = markers.copy()
            c[c != i] = 0
            c[c!=0]=255
            size=np.count_nonzero(c)
            [py,px] = ndimage.measurements.center_of_mass(c)
            x, y, w, h = cv2.boundingRect(c.astype(np.uint8))
            aspect_ratio = float(w) / h
            meanC=self.meanColor(self.Mask(frame,c.astype(np.uint8)))
            blob = [[int(px),int(py)], aspect_ratio, size, meanC]
            flatblob=self.flat(blob.copy())
            sim=self.simi(np.array(flatblob),np.array(flathand))
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
        if maxSim is not None and maxSim<self.MAX_SIM_THRESHOLD:
            if maxSim>=0.7*self.MAX_SIM_THRESHOLD:
                self.colorUpdateFlag=True
            self.info2.append(maxSim)
            print(maxSim)
            return handMask.astype(np.uint8),handBlob,True,BRec
        print(-1)
        self.info2.append(-100)
        return None,None,False,None
    def colorRangeindexes(self,gray):
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
    def clear_background(self,skinMask,moveMask):
        andMask=cv2.bitwise_and(skinMask,moveMask)
        andMask = cv2.medianBlur(andMask, self.clear_background_MEDIAN_BLUR)
        andMask = cv2.erode(andMask, self.kernel, iterations=self.clear_background_ITERATIONS)
        andMask = cv2.dilate(andMask, self.kernel, iterations=self.clear_background_ITERATIONS)
        self.fillMask(andMask)
        cv2.imshow("and",andMask)
        return andMask
    def getHandMask(self,skinMask,moveMask,frame):
        clearedBackground=self.clear_background(skinMask,moveMask)
        handMask, handblob, sucsses, BRec = self.findHand(clearedBackground, frame)
        if sucsses:
            self.addInfo(self.flat(self.hand.copy()))
            self.last_box = BRec
            self.hand = handblob
            return handMask
        return None
    def updateColors(self,YCbCr_image,HSV_image,Lab_image,mask,gray):
        maxrange, minrange = self.colorRangeindexes(self.Mask(gray, mask))
        self.lower_YCbCr_values = (YCbCr_image[minrange[0]][minrange[1]] * self.lower_color_extension).clip(self.lower_YCbCr_valueso, 255)
        self.upper_YCbCr_values = (YCbCr_image[maxrange[0]][maxrange[1]] * self.upper_color_extension).clip(0, self.upper_YCbCr_valueso)
        self.lower_Lab_values = (Lab_image[minrange[0]][minrange[1]] * self.lower_color_extension).clip(self.lower_Lab_valueso, 255)
        self.upper_Lab_values = (Lab_image[maxrange[0]][maxrange[1]] * self.upper_color_extension).clip(0, self.upper_Lab_valueso)
        self.lower_HSV_values = (HSV_image[minrange[0]][minrange[1]] * self.lower_color_extension).clip(self.lower_HSV_valueso, 255)
        self.upper_HSV_values = (HSV_image[maxrange[0]][maxrange[1]] * self.upper_color_extension).clip(0, self.upper_HSV_values)
    def color_segmentation(self,img,mask=None,gray=None):
        blured = cv2.GaussianBlur(img, (self.color_segmentation_IMG_GAUSSIAN_BLUR, self.color_segmentation_IMG_GAUSSIAN_BLUR), 0)
        YCbCr_image = cv2.cvtColor(blured, cv2.COLOR_BGR2YCrCb)
        HSV_image = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)
        Lab_image = cv2.cvtColor(blured, cv2.COLOR_BGR2Lab)
        if mask is not None and self.colorUpdateFlag:
            self.updateColors(YCbCr_image,HSV_image,Lab_image,mask,gray)
            self.colorUpdateFlag=False

        mask_YCbCr = cv2.inRange(YCbCr_image, self.lower_YCbCr_values, self.upper_YCbCr_values)
        mask_HSV = cv2.inRange(HSV_image, self.lower_HSV_values, self.upper_HSV_values)
        mask_Lab = cv2.inRange(Lab_image, self.lower_Lab_values, self.upper_Lab_values)
        skinMask=cv2.add(mask_HSV, mask_YCbCr, mask_Lab)

        skinMask = cv2.medianBlur(skinMask, self.color_segmentation_MEDIAN_BLUR)
        skinMask = cv2.erode(skinMask, self.kernel)
        skinMask = cv2.dilate(skinMask, self.kernel,iterations=self.color_segmentation_ITERATIONS)
        skinMask = cv2.erode(skinMask, self.kernel,iterations=self.color_segmentation_ITERATIONS)
        skinMask=cv2.GaussianBlur(skinMask, (self.color_segmentation_SKIN_MASK_GAUSSIAN_BLUR, self.color_segmentation_SKIN_MASK_GAUSSIAN_BLUR), 0)
        self.fillMask(skinMask)
        return skinMask
    def cropImage(self,image,Brec):
        return image[Brec[1]:Brec[1] + Brec[3],Brec[0] :Brec[0] + Brec[2]]
    def extractAngle(self,handMask):
        #croped=cropImage(handMask,last_box)
        lines = cv2.HoughLines(handMask, 1, np.pi / 180, threshold=self.threshold)
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
    def sendAngle(self,ang):
        if ang is not None:
            if math.fabs(ang-self.previousAngle)>5:
                # print(ang)
                self.roboRot(ang)
                self.previousAngle =ang
                ang=str(self.cameraNum)+","+str(ang)
                data=ang.encode('ascii')
                self.client.send(data)
    def restart(self):
        if self.handOnScene==1:
            self.colorUpdateFlag = True
            self.handOnScene = 0
            self.hand[self.handTitles["loc"]] = self.handOrigin
            self.roboRot(90)
            self.last_box = self.start_box.copy()
            self.hand = deepcopy(self.startHand)


    def main(self):
        self.roboRot(90)
        handMask=self.start.copy()
        while True:
            (grabbed, frame) = self.camera.read()
            moveMask=self.MovementMask(frame,self.Mask(frame,self.color_segmentation(frame)))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (int(0.9*x), int(0.7*y)),( int((x + w)), int(1.2*(y + h))), (255, 255, 0), cv2.FILLED)
            if self.handOnScene==1 and self.colorUpdateFlag:
                skinMask = self.color_segmentation(frame,handMask,gray)
            else:
                skinMask=self.color_segmentation(frame)
            skin = self.Mask(frame,skinMask)
            if self.handOnScene==1:
                self.frameCount+=1
                cv2.putText(frame, str(self.frameCount), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
                handMask = self.getHandMask(skinMask,moveMask,frame)
                if handMask is not None:
                    cv2.circle(frame, (self.hand[self.handTitles["loc"]][0], self.hand[self.handTitles["loc"]][1]), 7, (0, 255, 255), -1)
                    cv2.rectangle(frame, (self.last_box[0], self.last_box[1]), (self.last_box[0] + self.last_box[2], self.last_box[1] + self.last_box[3]), (255, 100, 0), 2)
                    ang=self.extractAngle(handMask)
                    self.sendAngle(ang)
                else:
                    self.restart()
            else:
                frame = cv2.add(frame, self.outline)
                currStartHand = self.Mask(self.Mask(frame,self.clear_background(skinMask,moveMask)), self.start)
                curr = np.count_nonzero(currStartHand)
                if curr >= self.outlineNum and cv2.waitKey(1) & 0xFF == ord("s"):
                    hand_color = cv2.bitwise_and(frame, frame, mask=self.start)
                    self.hand[self.handTitles["mean_color"]]=self.meanColor(hand_color)
                    self.hand[self.handTitles["size"]]=curr
                    self.hand[self.handTitles["mean_color"]]=self.meanColor(hand_color)
                    currStartHand_gray=cv2.cvtColor(currStartHand, cv2.COLOR_BGR2GRAY)
                    [py, px] = ndimage.measurements.center_of_mass(currStartHand_gray)
                    self.hand[self.handTitles["loc"]]=[py,px]
                    x, y, w, h = cv2.boundingRect(currStartHand_gray)
                    aspect_ratio = float(w) / h
                    self.hand[self.handTitles["aspect_ratio"]]=aspect_ratio
                    self.handOnScene = 1

            self.out.write(frame)
            # cv2.imshow("images", np.hstack((frame, skin)))
            cv2.imshow("images", frame)
            cv2.imshow("sdfgser", skin)

            key=cv2.waitKey(1)
            if  key& 0xFF == ord("q"):
                break
            if key & 0xFF == ord("r"):
                self.restart()

        self.camera.release()
        cv2.destroyAllWindows()
        client.close()
        for i in range(len(self.info)):
            plt.subplot(8, 1, i+1)
            plt.plot(self.info[i], label=self.titles[i])
            plt.legend()
        plt.subplot(8, 1, 8)
        plt.plot(self.info2, label="max sim")
        plt.legend()
        plt.show()
def setupServer():
    global client
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip,port))
    print("server started")
    server_socket.listen(1)
    print("waiting...")
    client, address = server_socket.accept()
    print("connected!")
setupServer()
cam=camera_angle(0,client)
threading.Thread(target = cam.main()).start()