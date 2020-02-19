import numpy as np
import cv2
from PIL import ImageGrab

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ", ", y)
        font = cv2.FONT_HERSHEY_COMPLEX
        strXY = str(x) + ", " + str(y)
        cv2.putText(frame, strXY, (x, y), font, 0.5, (255,255,0),2)
        cv2.imshow("frame", frame)

img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
img_np = np.array(img)
frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
cv2.fillPoly(frame, np.array([[ (1025, 650),(900, 650), (300,950), (1600,950)]]), 255)

cv2.imshow("frame", frame)
cv2.setMouseCallback("frame", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
