from time import sleep
import cv2
import numpy as np
camera=cv2.VideoCapture(0)
x=np.array([[-1,0,1],
            [-2,0,2],
            [-1,0,-1]],np.int32)

y=np.array([[-1,-2,-1],
            [0,0,0],
            [1,2,1]],np.int32)
dl=np.array([[0,-1,-2],
            [1,0,-1],
            [2,1,0]],np.int32)

dr=np.array([[-2,-1,0],
            [-1,0,1],
            [0,1,2]],np.int32)
def edges(im):
    edgesx = cv2.filter2D(im, -1, x)
    edgesy = cv2.filter2D(im, -1, y)
    edgesdl = cv2.filter2D(im, -1, dl)
    edgesdr = cv2.filter2D(im, -1, dr)
    edgesx = np.abs(edgesx)
    edgesy = np.abs(edgesy)
    edgesdl = np.abs(edgesdl)
    edgesdr = np.abs(edgesdr)
    return edgesy + edgesx + edgesdl + edgesdr
while True:
    (grabbed, frame) = camera.read()
    frame2=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
   # frame2=cv2.resize(frame2,(320,240))
    edge=edges(frame2[:, :, 0])+edges(frame2[:, :, 1])+edges(frame2[:, :, 2])
    edge=edge/3
    edge=edge.astype(np.uint8)
    cv2.imshow('frame',edge)
    cv2.imshow('frvame',frame2[:,:,2])
    #sleep(1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()