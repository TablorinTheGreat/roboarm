import win32api, win32con
import socket
import threading
from copy import deepcopy
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage

lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
upper_HSV_values = np.array([25, 255, 255], dtype="uint8")

# dynamic color ranges for YCbCr space
lower_YCbCr_values = np.array((0, 138, 67), dtype="uint8")
upper_YCbCr_values = np.array((255, 173, 133), dtype="uint8")

# dynamic color ranges for Lab space
lower_Lab_values = np.array([35, 135, 120], dtype="uint8")
upper_Lab_values = np.array([200, 180, 155], dtype="uint8")

# constant color ranges for hsv space
lower_HSV_valueso = np.array([0, 40, 0], dtype="uint8")
upper_HSV_valueso = np.array([25, 255, 255], dtype="uint8")

# constant color ranges for YCbCr space
lower_YCbCr_valueso = np.array((0, 138, 67), dtype="uint8")
upper_YCbCr_valueso = np.array((255, 173, 133), dtype="uint8")

# constant color ranges for Lab space
lower_Lab_valueso = np.array([35, 135, 120], dtype="uint8")
upper_Lab_valueso = np.array([200, 180, 155], dtype="uint8")

# flag for doing a color range update
colorUpdateFlag = True

# extensions for the dynamic color ranges
lower_color_extension = 0.5
upper_color_extension = 1.5

# the similarity threshold above this value the hand is considered to be out of scene
MAX_SIM_THRESHOLD = 4000

# weights for hand features
weights = [5, 5, 2, 3, 1, 1, 1]

# starting hand center of weight
handOrigin = [195, 250]
outline = cv2.imread("startOutline.jpg")
start = cv2.imread("start.png", cv2.IMREAD_GRAYSCALE)
# starting hand bounding box
start_box = [131, 96, 158, 324]
# 90% of the starting hands size
startingThreshold = 18806 * 0.9
# starting hand feature vector
startHand = [handOrigin, 0.487654, 1458, None]


camera = cv2.VideoCapture(0)


# a kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

# haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haar_cascade_frontaface.xml')

# lists that save the hand features for plots
handTitles = {"loc": 0, "aspect_ratio": 1, "size": 2, "mean_color": 3}
# the hand feature vector
hand = deepcopy(startHand)
# last bounding box of the hand
last_box = start_box.copy()

# is the hand on the scene flag
handOnScene = False

# grabbing the first frame and coloring the starting hand area with gray
(grabbed, frame) = camera.read()
cv2.rectangle(frame, (last_box[0], last_box[1]), (last_box[0] + last_box[2], last_box[1] + last_box[3]),
              (50, 50, 50), cv2.FILLED)
# applying the first frame to the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
fgbg.apply(frame)
# creating the skin color background subtractor
skinbg = cv2.createBackgroundSubtractorMOG2(varThreshold=6, detectShadows=False)
# blending parameter for the hand
stopped = 0.7
# black background for creating masks
bg = np.zeros((480, 640), np.uint8)

frameCount = 0

# clear background function morphological operation iterations constant
clear_background_ITERATIONS = 2
# clear background function median blur neighborhood size
clear_background_MEDIAN_BLUR = 7

# color segmentation function morphological operation iterations constant
color_segmentation_ITERATIONS = 4
# color segmentation function gaussian blur size
color_segmentation_IMG_GAUSSIAN_BLUR = 7
# color segmentation function median blur neighborhood size
color_segmentation_MEDIAN_BLUR = 7
# color segmentation function gaussian blur size for the skin mask
color_segmentation_SKIN_MASK_GAUSSIAN_BLUR = 5

# number of convexity defects of the last 20 frames
numCornerVotes=[5]*20
# is the hand open flag
isHandOpened=True

xfactor=0
yfactor=0

def initFactors():
    global xfactor,yfactor
    Width = win32api.GetSystemMetrics(0)
    Height = win32api.GetSystemMetrics(1)
    Fheight, Fwidth, channels = frame.shape
    xfactor=Width/Fwidth
    yfactor=Height/Fheight

def moveCursor(x,y):
    win32api.SetCursorPos((int(x*xfactor),int(y*yfactor)))
    
def click(x,y):
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, int(x*xfactor), int(y*yfactor), 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, int(x*xfactor), int(y*yfactor), 0, 0)


# function to flatten the hand feature vector to a single dimension
def flat(lst):
    lst[0:1]=lst[0]
    lst[4:5]=lst[4]
    return lst

# function to fill the holes in a mask
def fillMask(mask):
    # finding all the contours in the image
    img, contour, hier = cv2.findContours(mask, cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
    # filling them with white
    cv2.drawContours(mask, contour, 0, 255, -1)

# function to apply a mask to an image
def Mask(img,msk):
    return cv2.bitwise_and(img,img,mask=msk)

# a function to find the mean color in a masked image
def meanColor(frame):
    return frame.sum(axis=(0, 1)) / np.count_nonzero(frame, axis=(0, 1))

# function to return a mask with the moving pixels
def MovementMask(frame,skin):
    global last_box,fgbg,skinbg
    # creating a mask of the bounding box of the hand from the last frame
    invcurr=  bg.copy()
    cv2.rectangle(invcurr, (last_box[0], last_box[1]), (last_box[0] + last_box[2], last_box[1] + last_box[3]), (255, 255, 255),cv2.FILLED)

    # creating the inverted mask
    curr=(255 - invcurr)

    #if the hand is present then create a composite image with the hand faded
    if handOnScene:
        background =   Mask(frame,curr).astype(np.uint8) + \
                 ( stopped *  Mask( fgbg.getBackgroundImage(),invcurr)).astype(np.uint8) + \
                 ((1 -  stopped) *  Mask(frame,invcurr)).astype(np.uint8)

        # creating a composite image with the hand faded but only with the skin colored background
        skinBackground =  Mask(skin, curr).astype(np.uint8) + \
                 ( stopped *  Mask( skinbg.getBackgroundImage(), invcurr)).astype(np.uint8) + \
                 ((1 -  stopped) *  Mask(skin, invcurr)).astype(np.uint8)
    else:

        # if the hand is not on scene then take the regular frame
        background=frame
        skinBackground=skin

    # apply the backgrounds
    fgbg.apply(background)
    skinbg.apply(skinBackground)

    # return the or product of the regular background subtraction and the skin background subtraction
    return cv2.bitwise_or(fgbg.apply(frame, learningRate=0),skinbg.apply(skin, learningRate=0))

# difference function between 2 feature vectors
def diff( p1, handVec):
    return np.sum(np.multiply(((np.abs(p1-handVec)/handVec)*100),weights))

# find the hand blob inside the andmask
def findHand(andmask,frame):
    global colorUpdateFlag
    minDiff=None
    handBlob=None
    handMask=None
    BRec=None

    # finding the connected components inside the andmask
    ret, markers = cv2.connectedComponents(andmask)

    # flatning the hand feature vector
    flathand=flat(hand.copy())

    # finding the component with the highest similarity to the hand feature vector
    for i in range(1, ret):

        # making a black and white mask from the component
        c = markers.copy()
        c[c != i] = 0
        c[c!=0]=255

        # getting the components features: size, center of mass, aspect ratio, mean color and bounding rectangle
        size=np.count_nonzero(c)
        [py,px] = ndimage.measurements.center_of_mass(c)
        x, y, w, h = cv2.boundingRect(c.astype(np.uint8))
        aspect_ratio = float(w) / h
        meanC=meanColor(Mask(frame,c.astype(np.uint8)))
        blob = [[int(px),int(py)], aspect_ratio, size, meanC]
        flatblob=flat(blob.copy())

        # getting the difference between the 2 vectors
        Diff=diff(np.array(flatblob), np.array(flathand))

        # if a hand wasnt found yet or the difference of this blob
        # to the hand vector is lower then the previous than update the variables
        if handBlob is None or Diff<minDiff:
            minDiff = Diff
            handMask = c
            handBlob=blob
            BRec=[x,y,w,h]

    # if a hand was found and its difference is lower then the difference threshold
    if minDiff is not None and minDiff<MAX_SIM_THRESHOLD:

        # if the difference is greater then 70% of the difference threshold than update the color update flag
        if minDiff>=0.7*MAX_SIM_THRESHOLD:
            colorUpdateFlag=True

        return handMask.astype(np.uint8),handBlob,True,BRec
    return None,None,False,None

# a function to get the indexes of the highest and lowest pixel intensity of a mask
def colorRangeindexes(gray):
    max=None
    min=None
    imin=None
    jmin=None
    imax=None
    jmax=None

    # for every pixel in the image
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):

            # if there isnt any initial max value yet
            if max==None:
                # and this pixel isnt black meaning it is part of the mask
                if gray[i][j]>0:
                    max=min=gray[i][j]
                    imin=imax=i
                    jmin=jmax=j
            else:
                # if the current pixel intensity is higher than the max then update max value
                # and indexes of max value
                if gray[i][j]>max:
                    max=gray[i][j]
                    imax=i
                    jmax=j

                # if the current pixel intensity is lower than the minimum and its
                #  not 0 meaning its part of the mask then update min value and indexes of min value
                if gray[i][j]<min and gray[i][j]>0:
                    min=gray[i][j]
                    imin=i
                    jmin=j

    # return the indexes of the lowest and highest intensity pixel
    return [imax,jmax],[imin,jmin]

# function to merge the skin mask and the movement mask and "clean" the background
def clear_background(skinMask,moveMask):
    # perform an and operation on the masks
    andMask=cv2.bitwise_and(skinMask,moveMask)

    # perform a median blur on the mask to clear out salt and pepper noises
    andMask = cv2.medianBlur(andMask, clear_background_MEDIAN_BLUR)

    # perform erosion several times and dilation several times to make all the small
    # blobs disappear
    andMask = cv2.erode(andMask, kernel, iterations=clear_background_ITERATIONS)
    andMask = cv2.dilate(andMask, kernel, iterations=clear_background_ITERATIONS)

    # fill all the holes in the mask
    fillMask(andMask)
    return andMask

# function to find the hand mask from the frame
def getHandMask(skinMask,moveMask,frame):
    global hand,last_box
    # merge masks and clear the background
    clearedBackground=clear_background(skinMask,moveMask)
    # find the hand in the cleared background
    handMask, handblob, sucsses, BRec = findHand(clearedBackground, frame)

    #  if the hand was found then add info to the
    # plotting data and update the last bounding box and the hand vector
    if sucsses:
        last_box = BRec
        hand = handblob
        # if hand was found return the hand mask
        return handMask
    # if the hand wasn't found return none
    return None

# function to update the color ranges
def updateColors( YCbCr_image, HSV_image, Lab_image, handMask, gray):
    global lower_YCbCr_values,upper_YCbCr_values,lower_HSV_values,upper_HSV_values,lower_Lab_values,upper_Lab_values,lower_color_extension,upper_color_extension

    # find the lowest and highest pixel intensities of the gray scale hand mask
    maxrange, minrange = colorRangeindexes(Mask(gray, handMask))

    # update the YCbCr lower value with the color of lowest intensity times the
    # lower color extension and limit the range between the original value and 255
    lower_YCbCr_values = (YCbCr_image[minrange[0]][minrange[1]] * lower_color_extension).clip(lower_YCbCr_valueso, 255)

    # update the YCbCr upper value with the color of highest intensity times the
    # upper color extension and limit the range between the 0 and original value
    upper_YCbCr_values = (YCbCr_image[maxrange[0]][maxrange[1]] * upper_color_extension).clip(0, upper_YCbCr_valueso)

    # update the Lab lower value with the color of lowest intensity times the
    # lower color extension and limit the range between the original value and 255
    lower_Lab_values = (Lab_image[minrange[0]][minrange[1]] * lower_color_extension).clip(lower_Lab_valueso, 255)

    # update the Lab upper value with the color of highest intensity times the
    # upper color extension and limit the range between the 0 and original value
    upper_Lab_values = (Lab_image[maxrange[0]][maxrange[1]] * upper_color_extension).clip(0, upper_Lab_valueso)

    # update the HSV lower value with the color of lowest intensity times the
    # lower color extension and limit the range between the original value and 255
    lower_HSV_values = (HSV_image[minrange[0]][minrange[1]] * lower_color_extension).clip(lower_HSV_valueso, 255)

    # update the HSV upper value with the color of highest intensity times the
    # upper color extension and limit the range between the 0 and original value
    upper_HSV_values = (HSV_image[maxrange[0]][maxrange[1]] * upper_color_extension).clip(0, upper_HSV_values)

# function to create a skin mask from the frame and update the colors if necessary
def color_segmentation( img, handMask=None, gray=None):
    global colorUpdateFlag
    # convert bgr image to YCbCr, HSV and Lab color spaces
    YCbCr_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    HSV_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # if there is a handmask and the update flag is up then update the color ranges
    if handMask is not None and colorUpdateFlag:

        # update the color ranges
        updateColors(YCbCr_image, HSV_image, Lab_image, handMask, gray)

        # turn the update flag off
        colorUpdateFlag=False

    # get a skin mask from YCbCr space
    mask_YCbCr = cv2.inRange(YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)

    # get a skin mask from HSV space
    mask_HSV = cv2.inRange(HSV_image, lower_HSV_values, upper_HSV_values)

    # get a skin mask from Lab space
    mask_Lab = cv2.inRange(Lab_image, lower_Lab_values, upper_Lab_values)

    # combine masks with add function
    skinMask=cv2.add(mask_HSV, mask_YCbCr, mask_Lab)

    # perform a median blur on the mask to clear out salt and pepper noises
    skinMask = cv2.medianBlur(skinMask, color_segmentation_MEDIAN_BLUR)

    # perform erosion to remove small blobs
    skinMask = cv2.erode(skinMask, kernel)

    # perform dilation and then erosion to close holes
    skinMask = cv2.dilate(skinMask, kernel,iterations=color_segmentation_ITERATIONS)
    skinMask = cv2.erode(skinMask, kernel,iterations=color_segmentation_ITERATIONS)
    return skinMask


# a function to restart the hand
def restart():
    global colorUpdateFlag,handOnScene,hand,last_box

    # if the hand is on scene
    if handOnScene:
        # reset the color update flag to true
        colorUpdateFlag = True

        # set the hand on scene flag to false
        handOnScene = False

        # reset the last bounding box
        last_box = start_box.copy()

        # reset the hand vector
        hand = deepcopy(startHand)

        # reset the corner voters
        numCornerVotes = [5] * 20

# a function to tell if the hand is open or not
def isHandOpen(handMask):
    # find the hand contour
    sdf, contours, hierarchy = cv2.findContours(handMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    # downsample the hands contour
    cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

    # find the hands convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)

    # find the defects in the convex hull
    defects = cv2.convexityDefects(cnt, hull)
    if defects is not None:
        # add the number of convexity defects to the list
        numCornerVotes.append(defects.shape[0])

        # remove the oldest voter
        del numCornerVotes[0]

        # get the mean of defects over the last 20 frames
        mean=np.array(numCornerVotes).mean()

        # if the mean is under 3 then the hand is closed
        if mean<3:
            return False
        else:
            return True
    # if the defects are none return the last value
    return isHandOpened

handMask=None
initFactors()
# applying the first frame to the skin background subtractor
skinbg.apply(Mask(frame,color_segmentation(frame)))
while True:
    # reading the frame
    (grabbed, frame) = camera.read()

    # getting the movement mask
    moveMask=MovementMask(frame,Mask(frame,color_segmentation(frame)))

    # getting the grayscale of the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # searching for faces with haar cascade
    faces = face_cascade.detectMultiScale(gray)
    # for each face draw over it with a colored rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (int(0.9*x), int(0.7*y)),( int((x + w)), int(1.2*(y + h))), (255, 255, 0), cv2.FILLED)

    # if the hand is on scene and the color update flag
    # is up then use the color segmentation with the update function
    if handOnScene and colorUpdateFlag:
        skinMask = color_segmentation(frame,handMask,gray)

    # else use it without the update
    else:
        skinMask=color_segmentation(frame)
    # if the hand is on scene
    if handOnScene:
        # increment the frame counter and show it on the frame
        frameCount+=1
        cv2.putText(frame, str(frameCount), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)

        # get the hand mask from the frame
        handMask = getHandMask(skinMask,moveMask,frame)

        # if the hand is in the frame
        if handMask is not None:
            # draw a circle over the hand's center of mass
            col=0
            if isHandOpened:
                col=(0, 255, 255)
            else:
                col=(0,0,255)
            cv2.circle(frame, (hand[handTitles["loc"]][0], hand[handTitles["loc"]][1]), 7, col, -1)
            moveCursor(hand[handTitles["loc"]][0],hand[handTitles["loc"]][1])
            # draw the hands bounding box
            cv2.rectangle(frame, (last_box[0], last_box[1]), (last_box[0] + last_box[2], last_box[1] + last_box[3]), (255, 100, 0), 2)

            # if this camera angle is topview and the hand's state has changed
            if isHandOpen(handMask)!= isHandOpened:
                # update the hand's state
                isHandOpened = not isHandOpened
                if isHandOpened:
                    click(hand[handTitles["loc"]][0],hand[handTitles["loc"]][1])


        # if the hand is not on scene then restart
        else:
            restart()
    else:
        # if the hand is not on scene then add the outline
        # of the start hand to the frame
        frame = cv2.add(frame, outline)

        # get the number of skin colored pixels that are moving
        # inside the starting hand outline
        cl=clear_background(skinMask,moveMask)
        cv2.imshow("fghjk",cl)
        currStartHand = Mask(Mask(frame,cl), start)
        curr = np.count_nonzero(currStartHand)

        # if the number is above the starting threshold and the user pressed s
        if curr >= startingThreshold and cv2.waitKey(1) & 0xFF == ord("s"):
            # get the masked frame
            hand_color = Mask(frame, start)

            # update the hands mean color
            hand[handTitles["mean_color"]]=meanColor(hand_color)

            # update the hand's size
            hand[handTitles["size"]]=curr

            # get the hands center of mass and update the hands vector
            currStartHand_gray=cv2.cvtColor(currStartHand, cv2.COLOR_BGR2GRAY)
            [py, px] = ndimage.measurements.center_of_mass(currStartHand_gray)
            hand[handTitles["loc"]]=[py,px]

            # get the hand's bounding box
            x, y, w, h = cv2.boundingRect(currStartHand_gray)

            # get the hand's aspect ratio
            aspect_ratio = float(w) / h

            # update the hand's last bounding box
            last_box = [x, y, w, h]

            # update the hand's aspect ratio
            hand[handTitles["aspect_ratio"]]=aspect_ratio

            # update the hand on scene flag to be true
            handOnScene = True

    # show the frame
    cv2.imshow("images", frame)

    # check user input
    key=cv2.waitKey(1)

    # if the user entered q then exit the program
    if key & 0xFF == ord("q"):
        break

    # if the user entered r then restart the program
    if key & 0xFF == ord("r"):
        restart()

# realese the camera
camera.release()

# close all opened windows
cv2.destroyAllWindows()
