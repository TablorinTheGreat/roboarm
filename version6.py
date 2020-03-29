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


# an object to represent a camera angle. we have sideview camera and topview camera
class camera_angle:

    # this function initiates the camera angle object and all of its variables and constants
    def __init__(self, cameraNum, client):

        # dynamic color ranges for hsv space
        self.lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
        self.upper_HSV_values = np.array([25, 255, 255], dtype="uint8")

        # dynamic color ranges for YCbCr space
        self.lower_YCbCr_values = np.array((0, 138, 67), dtype="uint8")
        self.upper_YCbCr_values = np.array((255, 173, 133), dtype="uint8")

        # dynamic color ranges for Lab space
        self.lower_Lab_values = np.array([35, 135, 120], dtype="uint8")
        self.upper_Lab_values = np.array([200, 180, 155], dtype="uint8")

        # constant color ranges for hsv space
        self.lower_HSV_valueso = np.array([0, 40, 0], dtype="uint8")
        self.upper_HSV_valueso = np.array([25, 255, 255], dtype="uint8")

        # constant color ranges for YCbCr space
        self.lower_YCbCr_valueso = np.array((0, 138, 67), dtype="uint8")
        self.upper_YCbCr_valueso = np.array((255, 173, 133), dtype="uint8")

        # constant color ranges for Lab space
        self.lower_Lab_valueso = np.array([35, 135, 120], dtype="uint8")
        self.upper_Lab_valueso = np.array([200, 180, 155], dtype="uint8")

        # flag for doing a color range update
        self.colorUpdateFlag = True

        # extensions for the dynamic color ranges
        self.lower_color_extension = 0.5
        self.upper_color_extension = 1.5

        # threshold for hough lines transform
        self.threshold = 60

        # hand's last angle to screen small angle changes
        self.previousAngle = 0

        # the similarity threshold above this value the hand is considered to be out of scene
        self.MAX_SIM_THRESHOLD = 4000

        # weights for hand features
        self.weights = [5, 5, 2, 3, 1, 1, 1]

        # starting hand center of weight
        self.handOrigin = [195, 250]
        self.outline = cv2.imread("startOutline.jpg")
        self.start = cv2.imread("start.png", cv2.IMREAD_GRAYSCALE)
        # starting hand bounding box
        self.start_box = [131, 96, 158, 324]
        # 90% of the starting hands size
        self.startingThreshold = 18806 * 0.9
        # starting hand feature vector
        self.startHand = [self.handOrigin, 0.487654, 1458, None]

        # camera number
        self.cameraNum = cameraNum
        self.camera = cv2.VideoCapture(self.cameraNum)

        # the tcp client connected to the robotic arm
        self.client = client

        # a kernel for morphological operations
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        # haar cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier('haar_cascade_frontaface.xml')

        # lists that save the hand features for plots
        self.titles = ["x", "y", "aspect ratio", "size", "r", "g", "b"]
        self.info = [[], [], [], [], [], [], []]
        self.info2 = []
        self.handTitles = {"loc": 0, "aspect_ratio": 1, "size": 2, "mean_color": 3}
        # the hand feature vector
        self.hand = deepcopy(self.startHand)
        # last bounding box of the hand
        self.last_box = self.start_box.copy()

        # is the hand on the scene flag
        self.handOnScene = False

        # grabbing the first frame and coloring the starting hand area with gray
        (grabbed, self.frame) = self.camera.read()
        cv2.rectangle(self.frame, (self.last_box[0], self.last_box[1]),
                      (self.last_box[0] + self.last_box[2], self.last_box[1] + self.last_box[3]),
                      (50, 50, 50), cv2.FILLED)
        # applying the first frame to the background subtractor
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.fgbg.apply(self.frame)
        # creating the skin color background subtractor
        self.skinbg = cv2.createBackgroundSubtractorMOG2(varThreshold=6, detectShadows=False)
        # blending parameter for the hand
        self.stopped = 0.7
        # black background for creating masks
        self.bg = np.zeros((480, 640), np.uint8)

        # video recorder for analyzing the session
        self.out = cv2.VideoWriter('lastOutput.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (640, 480))
        self.frameCount = 0

        # clear background function morphological operation iterations constant
        self.clear_background_ITERATIONS = 2
        # clear background function median blur neighborhood size
        self.clear_background_MEDIAN_BLUR = 7

        # color segmentation function morphological operation iterations constant
        self.color_segmentation_ITERATIONS = 4
        # color segmentation function gaussian blur size
        self.color_segmentation_IMG_GAUSSIAN_BLUR = 7
        # color segmentation function median blur neighborhood size
        self.color_segmentation_MEDIAN_BLUR = 7
        # color segmentation function gaussian blur size for the skin mask
        self.color_segmentation_SKIN_MASK_GAUSSIAN_BLUR = 5

        # number of convexity defects of the last 20 frames
        self.numCornerVotes = [5] * 20
        # is the hand open flag
        self.isHandOpened = True
        # claw constant for client to recognize the values
        self.CLAW_NUM = 3
        # is the current camera angle is top camera or not
        self.isTopCamera = cameraNum == 0

    # function to flatten the hand feature vector to a single dimension
    def flat(self, lst):
        lst[0:1] = lst[0]
        lst[4:5] = lst[4]
        return lst

    # function to add the hand feature vector to the info list to plot
    def addInfo(self, hand):
        for i in range(len(hand)):
            self.info[i].append(hand[i])

    # function to fill the holes in a mask
    def fillMask(self, mask):
        # finding all the contours in the image
        img, contour, hier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # filling them with white
        cv2.drawContours(mask, contour, 0, 255, -1)

    # function to apply a mask to an image
    def Mask(self, img, msk):
        return cv2.bitwise_and(img, img, mask=msk)

    # function to find the mean color of an image

    # a function to find the mean color in a masked image
    def meanColor(self, frame):
        return frame.sum(axis=(0, 1)) / np.count_nonzero(frame, axis=(0, 1))

    # function to return a mask with the moving pixels
    def MovementMask(self, frame, skin):
        # creating a mask of the bounding box of the hand from the last frame
        invcurr = self.bg.copy()
        cv2.rectangle(invcurr, (self.last_box[0], self.last_box[1]),
                      (self.last_box[0] + self.last_box[2], self.last_box[1] + self.last_box[3]), (255, 255, 255),
                      cv2.FILLED)

        # creating the inverted mask
        curr = (255 - invcurr)

        # if the hand is present then create a composite image with the hand faded
        if self.handOnScene:
            background = self.Mask(frame, curr).astype(np.uint8) + \
                         (self.stopped * self.Mask(self.fgbg.getBackgroundImage(), invcurr)).astype(np.uint8) + \
                         ((1 - self.stopped) * self.Mask(frame, invcurr)).astype(np.uint8)

            # creating a composite image with the hand faded but only with the skin colored background
            skinBackground = self.Mask(skin, curr).astype(np.uint8) + \
                             (self.stopped * self.Mask(self.skinbg.getBackgroundImage(), invcurr)).astype(np.uint8) + \
                             ((1 - self.stopped) * self.Mask(skin, invcurr)).astype(np.uint8)
        else:

            # if the hand is not on scene then take the regular frame
            background = frame
            skinBackground = skin

        # apply the backgrounds
        self.fgbg.apply(background)
        self.skinbg.apply(skinBackground)

        # return the or product of the regular background subtraction and the skin background subtraction
        return cv2.bitwise_or(self.fgbg.apply(frame, learningRate=0), self.skinbg.apply(skin, learningRate=0))

    # similarity function between 2 feature vectors
    def simi(self, p1, handVec):
        return np.sum(np.multiply(((np.abs(p1 - handVec) / handVec) * 100), self.weights))

    # find the hand blob inside the andmask
    def findHand(self, andmask, frame):
        minDiff = None
        handBlob = None
        handMask = None
        BRec = None

        # finding the connected components inside the andmask
        ret, markers = cv2.connectedComponents(andmask)

        # flatning the hand feature vector
        flathand = self.flat(self.hand.copy())

        # finding the component with the highest similarity to the hand feature vector
        for i in range(1, ret):

            # making a black and white mask from the component
            c = markers.copy()
            c[c != i] = 0
            c[c != 0] = 255

            # getting the components features: size, center of mass, aspect ratio, mean color and bounding rectangle
            size = np.count_nonzero(c)
            [py, px] = ndimage.measurements.center_of_mass(c)
            x, y, w, h = cv2.boundingRect(c.astype(np.uint8))
            aspect_ratio = float(w) / h
            meanC = self.meanColor(self.Mask(frame, c.astype(np.uint8)))
            blob = [[int(px), int(py)], aspect_ratio, size, meanC]
            flatblob = self.flat(blob.copy())

            # getting the difference between the 2 vectors
            diff = self.simi(np.array(flatblob), np.array(flathand))
            if handBlob is None or diff < minDiff:
                minDiff = diff
                handMask = c
                handBlob = blob
                BRec = [x, y, w, h]
        if minDiff is not None and minDiff < self.MAX_SIM_THRESHOLD:
            if minDiff >= 0.7 * self.MAX_SIM_THRESHOLD:
                self.colorUpdateFlag = True
            self.info2.append(minDiff)
            # print(minDiff)
            return handMask.astype(np.uint8), handBlob, True, BRec
        return None, None, False, None

    # a function to get the indexes of the highest and lowest pixel intensity of a mask
    def colorRangeindexes(self, gray):
        max = None
        min = None
        imin = None
        jmin = None
        imax = None
        jmax = None

        # for every pixel in the image
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):

                # if there isnt any initial max value yet
                if max == None:
                    # and this pixel isnt black meaning it is part of the mask
                    if gray[i][j] > 0:
                        max = min = gray[i][j]
                        imin = imax = i
                        jmin = jmax = j
                else:
                    # if the current pixel intensity is higher than the max then update max value
                    # and indexes of max value
                    if gray[i][j] > max:
                        max = gray[i][j]
                        imax = i
                        jmax = j

                    # if the current pixel intensity is lower than the minimum and its
                    #  not 0 meaning its part of the mask then update min value and indexes of min value
                    if gray[i][j] < min and gray[i][j] > 0:
                        min = gray[i][j]
                        imin = i
                        jmin = j

        # return the indexes of the lowest and highest intensity pixel
        return [imax, jmax], [imin, jmin]

    # function to merge the skin mask and the movement mask and "clean" the background
    def clear_background(self, skinMask, moveMask):
        # perform an and operation on the masks
        andMask = cv2.bitwise_and(skinMask, moveMask)

        # perform a median blur on the mask to clear out salt and pepper noises
        andMask = cv2.medianBlur(andMask, self.clear_background_MEDIAN_BLUR)

        # perform erosion several times and dilation several times to make all the small
        # blobs disappear
        andMask = cv2.erode(andMask, self.kernel, iterations=self.clear_background_ITERATIONS)
        andMask = cv2.dilate(andMask, self.kernel, iterations=self.clear_background_ITERATIONS)

        # fill all the holes in the mask
        self.fillMask(andMask)
        return andMask

    # function to find the hand mask from the frame
    def getHandMask(self, skinMask, moveMask, frame):
        # merge masks and clear the background
        clearedBackground = self.clear_background(skinMask, moveMask)

        # find the hand in the cleared background
        handMask, handblob, sucsses, BRec = self.findHand(clearedBackground, frame)

        #  if the hand was found then add info to the
        # plotting data and update the last bounding box and the hand vector
        if sucsses:
            self.addInfo(self.flat(self.hand.copy()))
            self.last_box = BRec
            self.hand = handblob
            # if hand was found return the hand mask
            return handMask
        # if the hand wasn't found return none
        return None

    # function to update the color ranges
    def updateColors(self, YCbCr_image, HSV_image, Lab_image, handMask, gray):
        # find the lowest and highest pixel intensities of the gray scale hand mask
        maxrange, minrange = self.colorRangeindexes(self.Mask(gray, handMask))

        # update the YCbCr lower value with the color of lowest intensity times the
        # lower color extension and limit the range between the original value and 255
        self.lower_YCbCr_values = (YCbCr_image[minrange[0]][minrange[1]] * self.lower_color_extension).clip(
            self.lower_YCbCr_valueso, 255)

        # update the YCbCr upper value with the color of highest intensity times the
        # upper color extension and limit the range between the 0 and original value
        self.upper_YCbCr_values = (YCbCr_image[maxrange[0]][maxrange[1]] * self.upper_color_extension).clip(0,
                                                                                                            self.upper_YCbCr_valueso)

        # update the Lab lower value with the color of lowest intensity times the
        # lower color extension and limit the range between the original value and 255
        self.lower_Lab_values = (Lab_image[minrange[0]][minrange[1]] * self.lower_color_extension).clip(
            self.lower_Lab_valueso, 255)

        # update the Lab upper value with the color of highest intensity times the
        # upper color extension and limit the range between the 0 and original value
        self.upper_Lab_values = (Lab_image[maxrange[0]][maxrange[1]] * self.upper_color_extension).clip(0,
                                                                                                        self.upper_Lab_valueso)

        # update the HSV lower value with the color of lowest intensity times the
        # lower color extension and limit the range between the original value and 255
        self.lower_HSV_values = (HSV_image[minrange[0]][minrange[1]] * self.lower_color_extension).clip(
            self.lower_HSV_valueso, 255)

        # update the HSV upper value with the color of highest intensity times the
        # upper color extension and limit the range between the 0 and original value
        self.upper_HSV_values = (HSV_image[maxrange[0]][maxrange[1]] * self.upper_color_extension).clip(0,
                                                                                                        self.upper_HSV_values)

    # function to create a skin mask from the frame and update the colors if necessary
    def color_segmentation(self, img, handMask=None, gray=None):
        # convert bgr image to YCbCr, HSV and Lab color spaces
        YCbCr_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        HSV_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        Lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        # if there is a handmask and the update flag is up then update the color ranges
        if handMask is not None and self.colorUpdateFlag:
            # update the color ranges
            self.updateColors(YCbCr_image, HSV_image, Lab_image, handMask, gray)

            # turn the update flag off
            self.colorUpdateFlag = False

        # get a skin mask from YCbCr space
        mask_YCbCr = cv2.inRange(YCbCr_image, self.lower_YCbCr_values, self.upper_YCbCr_values)

        # get a skin mask from HSV space
        mask_HSV = cv2.inRange(HSV_image, self.lower_HSV_values, self.upper_HSV_values)

        # get a skin mask from Lab space
        mask_Lab = cv2.inRange(Lab_image, self.lower_Lab_values, self.upper_Lab_values)

        # combine masks with add function
        skinMask = cv2.add(mask_HSV, mask_YCbCr, mask_Lab)

        # perform a median blur on the mask to clear out salt and pepper noises
        skinMask = cv2.medianBlur(skinMask, self.color_segmentation_MEDIAN_BLUR)

        # perform erosion to remove small blobs
        skinMask = cv2.erode(skinMask, self.kernel)

        # perform dilation and then erosion to close holes
        skinMask = cv2.dilate(skinMask, self.kernel, iterations=self.color_segmentation_ITERATIONS)
        skinMask = cv2.erode(skinMask, self.kernel, iterations=self.color_segmentation_ITERATIONS)
        return skinMask

    # function to get the angle in which the hand is positioned
    def extractAngle(self, handMask):

        # use the hough lines transform to find the line of the hand
        lines = cv2.HoughLines(handMask, 1, np.pi / 180, threshold=self.threshold)
        if lines is not None:
            # take the line with most voters
            theta = lines[0][0][1]
            cos = np.cos(theta)
            sin = np.sin(theta)

            # calculate the slope of the line
            m = -cos / sin

            # calculate the degrees of the line
            ang = math.degrees(math.atan(m))

            # convert from range of 0 to -90 to the range of 0 to 180
            ang = -ang
            if ang < 0:
                ang = 180 + ang
            return ang

    # a function to send the angle of the hand to the robotic arm
    def sendAngle(self, ang):
        # if the angle is not none and there was a change in angle greater than 5 degrees
        if ang is not None:
            if math.fabs(ang - self.previousAngle) > 5:
                # update the previous angle
                self.previousAngle = ang

                # send the angle with the camera num
                ang = str(self.cameraNum) + "," + str(ang)
                self.sendData(ang)

    # a function to send data over the socket to the robotic arm
    def sendData(self, data):
        # encode the string and send it
        data = data.encode('ascii')
        self.client.send(data)

    # a function to restart the hand
    def restart(self):
        # if the hand is on scene
        if self.handOnScene:
            # reset the color update flag to true
            self.colorUpdateFlag = True

            # set the hand on scene flag to false
            self.handOnScene = False

            # reset the last bounding box
            self.last_box = self.start_box.copy()

            # reset the hand vector
            self.hand = deepcopy(self.startHand)

            # reset the corner voters
            self.numCornerVotes = [5] * 20

    # a function to tell if the hand is open or not
    def isHandOpen(self, handMask):
        # find the hand contour
        sdf, contours, hierarchy = cv2.findContours(handMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]

        # downsample the hands contour
        cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

        # find the hands convex hull
        hull = cv2.convexHull(cnt, returnPoints=False)

        # find the defects in the convex hull
        defects = cv2.convexityDefects(cnt, hull)
        if defects is not None:
            # add the number of convexity defects to the list
            self.numCornerVotes.append(defects.shape[0])

            # remove the oldest voter
            del self.numCornerVotes[0]

            # get the mean of defects over the last 20 frames
            mean = np.array(self.numCornerVotes).mean()

            # if the mean is under 3 then the hand is closed
            if mean < 3:
                return False
            else:
                return True
        # if the defects are none return the last value
        return self.isHandOpened

    def main(self):
        handMask = None

        # applying the first frame to the skin background subtractor
        self.skinbg.apply(self.Mask(self.frame, self.color_segmentation(self.frame)))
        while True:
            # reading the frame
            (grabbed, self.frame) = self.camera.read()

            # getting the movement mask
            moveMask = self.MovementMask(self.frame, self.Mask(self.frame, self.color_segmentation(self.frame)))

            # getting the grayscale of the frame
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            # searching for faces with haar cascade
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            # for each face draw over it with a colored rectangle
            for (x, y, w, h) in faces:
                cv2.rectangle(self.frame, (int(0.9 * x), int(0.7 * y)), (int((x + w)), int(1.2 * (y + h))),
                              (255, 255, 0), cv2.FILLED)

            # if the hand is on scene and the color update flag
            # is up then use the color segmentation with the update function
            if self.handOnScene and self.colorUpdateFlag:
                skinMask = self.color_segmentation(self.frame, handMask, gray)

            # else use it without the update
            else:
                skinMask = self.color_segmentation(self.frame)

            # if the hand is on scene
            if self.handOnScene:
                # increment the frame counter and show it on the frame
                self.frameCount += 1
                cv2.putText(self.frame, str(self.frameCount), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255),
                            2, cv2.LINE_AA)

                # get the hand mask from the frame
                handMask = self.getHandMask(skinMask, moveMask, self.frame)

                # if the hand is in the frame
                if handMask is not None:
                    # draw a circle over the hand's center of mass
                    cv2.circle(self.frame, (self.hand[self.handTitles["loc"]][0], self.hand[self.handTitles["loc"]][1]),
                               7, (0, 255, 255), -1)

                    # draw the hands bounding box
                    cv2.rectangle(self.frame, (self.last_box[0], self.last_box[1]),
                                  (self.last_box[0] + self.last_box[2], self.last_box[1] + self.last_box[3]),
                                  (255, 100, 0), 2)

                    # extract the hands angle ad send it
                    ang = self.extractAngle(handMask)
                    self.sendAngle(ang)

                    # if this camera angle is topview and the hand's state has changed
                    if self.isTopCamera and self.isHandOpen(handMask) != self.isHandOpened:
                        # update the hand's state
                        self.isHandOpened = not self.isHandOpened

                        # send the closing or opening command to the robotic arm
                        data = str(self.CLAW_NUM) + "," + str(self.isHandOpened)
                        self.sendData(data)

                # if the hand is not on scene then restart
                else:
                    self.restart()
            else:
                # if the hand is not on scene then add the outline
                # of the start hand to the frame
                self.frame = cv2.add(self.frame, self.outline)

                # get the number of skin colored pixels that are moving
                # inside the starting hand outline
                currStartHand = self.Mask(self.Mask(self.frame, self.clear_background(skinMask, moveMask)), self.start)
                curr = np.count_nonzero(currStartHand)

                # if the number is above the starting threshold and the user pressed s
                if curr >= self.startingThreshold and cv2.waitKey(1) & 0xFF == ord("s"):
                    # get the masked frame
                    hand_color = self.Mask(self.frame, self.start)

                    # update the hands mean color
                    self.hand[self.handTitles["mean_color"]] = self.meanColor(hand_color)

                    # update the hand's size
                    self.hand[self.handTitles["size"]] = curr

                    # get the hands center of mass and update the hands vector
                    currStartHand_gray = cv2.cvtColor(currStartHand, cv2.COLOR_BGR2GRAY)
                    [py, px] = ndimage.measurements.center_of_mass(currStartHand_gray)
                    self.hand[self.handTitles["loc"]] = [py, px]

                    # get the hand's bounding box
                    x, y, w, h = cv2.boundingRect(currStartHand_gray)

                    # get the hand's aspect ratio
                    aspect_ratio = float(w) / h

                    # update the hand's last bounding box
                    self.last_box = [x, y, w, h]

                    # update the hand's aspect ratio
                    self.hand[self.handTitles["aspect_ratio"]] = aspect_ratio

                    # update the hand on scene flag to be true
                    self.handOnScene = True

            # record the frame
            self.out.write(self.frame)

            # show the frame
            cv2.imshow("images", self.frame)

            # check user input
            key = cv2.waitKey(1)

            # if the user entered q then exit the program
            if key & 0xFF == ord("q"):
                break

            # if the user entered r then restart the program
            if key & 0xFF == ord("r"):
                self.restart()

        # realese the camera
        self.camera.release()

        # close all opened windows
        cv2.destroyAllWindows()

        # close the connection to the robotic arm
        self.client.close()

        # release the video writer
        self.out.release()

        # plot each feature in the hand's vector
        for i in range(len(self.info)):
            plt.subplot(8, 1, i + 1)
            plt.plot(self.info[i], label=self.titles[i])
            plt.legend()

        # plot the difference of the hand
        plt.subplot(8, 1, 8)
        plt.plot(self.info2, label="diff")
        plt.legend()

        # show the plots
        plt.show()


# a function to setup a server
def setupServer(ip, port):
    global client
    # create a server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # bind the socket
    server_socket.bind((ip, port))
    print("server started")

    # listen for a client
    server_socket.listen(1)
    print("waiting...")

    # accept the client
    client, address = server_socket.accept()
    print("connected!")


# setup a server
setupServer(ip, port)

# create a camera angle object with the first camera
cam0 = camera_angle(0, client)

# start the hand program on a different thread
threading.Thread(target=cam0.main()).start()

# create a camera angle object with the second camera
cam1 = camera_angle(1, client)

# start the hand program on a different thread
# threading.Thread(target = cam1.main()).start()