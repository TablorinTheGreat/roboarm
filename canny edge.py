import cv2
import numpy as np;
x=laplacian = np.array((
	[-1, 0,1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")
y=laplacian = np.array((
	[1, 2,1],
	[0, 0, 0],
	[-1, -2, -1]), dtype="int")
# Read image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
camera = cv2.VideoCapture(0)
while True:
    (g,im)=camera.read()
    im2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    t = cv2.Canny(im2, 10, 200)
    cv2.imshow("i",t)
    im2, contours, hierarchy = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, contours, -1, (0, 255, 0), 2)
    cv2.imshow("im",im)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()