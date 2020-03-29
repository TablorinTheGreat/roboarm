import cv2
import numpy as np
camera=cv2.VideoCapture(0)
a=np.array([[0,0,0],
            [4,0,-4],
            [0,0,0]],np.int32)

b=np.array([[0,4,0],
            [0,0,0],
            [0,-4,0]],np.int32)
c=np.array([[4,0,0],
            [0,0,0],
            [0,0,-4]],np.int32)

d=np.array([[0,0,-4],
            [0,0,0],
            [4,0,0]],np.int32)

x=np.array([[0,-1,0],
            [0,2,0],
            [0,-1,0]],np.int32)

y=np.array([[0,0,0],
            [-1,2,-1],
            [0,0,0]],np.int32)
while True:
    (grabbed, frame) = camera.read()
    #frame=cv2.resize(frame,(320,240))
    frame= cv2.medianBlur(frame, 9)
    image=2*frame[:,:,0]+4*frame[:,:,1]+3*frame[:,:,2]
    edgesx = cv2.filter2D(image, -1, a)
    edgesy = cv2.filter2D(image, -1, b)
    edgesdl = cv2.filter2D(image, -1, c)
    edgesdr = cv2.filter2D(image, -1, d)
    image=np.max(np.array([edgesx,edgesy,edgesdl,edgesdr]),axis=0)
    thresh=np.sum(image)/np.count_nonzero(image)
    ret, thresh1=cv2.threshold(image,thresh,255,cv2.THRESH_BINARY)
    thresh1 = cv2.filter2D(thresh1, -1, x)
    thresh1 = cv2.filter2D(thresh1, -1, y)
    cv2.imshow("s   adfs",thresh1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()