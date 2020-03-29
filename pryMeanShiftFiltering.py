import cv2
import numpy as np
c=cv2.VideoCapture(0)
CANNY_THRESH_1 = 100
CANNY_THRESH_2 = 200
while True:
    (g,img)=c.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1= cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    img=cv2.pyrMeanShiftFiltering(img, 20, 45, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    cv2.imshow("s",np.hstack((img1,img2)))
    cv2.imshow("sasd",img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
c.release()
cv2.destroyAllWindows()
