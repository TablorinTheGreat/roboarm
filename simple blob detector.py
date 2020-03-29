import cv2
import numpy as np
im = cv2.imread("blob.png", cv2.IMREAD_GRAYSCALE)
im=255-im
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 10
params.maxThreshold = 200
params.filterByArea = False
params.minArea = 1500
params.filterByCircularity = False
params.minCircularity = 0.1
params.filterByConvexity = False
params.minConvexity = 0.87
params.filterByInertia = False
params.minInertiaRatio = 0.01
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(im)
colorUpdateInterval, d=detector.compute(im, keypoints)
im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)