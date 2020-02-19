import cv2
import numpy as np

def nothing(x):
    pass

def edgeDetection(frame, blurSize, lowT, highT):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurredFrame = cv2.GaussianBlur(grayFrame, (blurSize, blurSize), 0)
    edgeDetectedFrame = cv2.Canny(blurredFrame, lowT, highT)
    return edgeDetectedFrame

cv2.namedWindow("trash")
cv2.createTrackbar("lowT", "trash", 0, 255, nothing)
cv2.createTrackbar("highT", "trash", 0, 255, nothing)

capture = cv2.VideoCapture('ontheroadagain.mp4')
    
while capture.isOpened():
    lowT = cv2.getTrackbarPos("lowT","trash")
    highT = cv2.getTrackbarPos("highT","trash")
    _, videoFrame = capture.read()
    edgeDetectedFrame = edgeDetection(videoFrame, 3, lowT, highT)
    cv2.imshow('edges', edgeDetectedFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()