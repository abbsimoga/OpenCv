import cv2
import numpy as np
from PIL import ImageGrab

def edgeDetection(frame, blurSize, lowT, highT):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurredFrame = cv2.GaussianBlur(grayFrame, (blurSize, blurSize), 0)
    edgeDetectedFrame = cv2.Canny(blurredFrame, lowT, highT)
    return edgeDetectedFrame

def makeCoordinates(frame, lines, priorCords):
    lineImage = np.zeros_like(frame)
    cords = []
    for line, priorLine in zip(lines, priorCords):
        if np.isnan(line).any():
            cords.append(priorLine)
            lineImage = drawRoadLines(lineImage, priorLine[:4], 2, 255, 255, 0)
        else:
            slope, intercept = line
            y1 = frame.shape[0]
            y2 = int(y1*(3.25/5))
            x1 = int((y1-intercept)/slope)
            x2 = int((y2-intercept)/slope)
            y3 = int(y1*(4/5))
            x3 = int((y3-intercept)/slope)
            cordsAverage = np.average(np.array([np.array([x1,y1,x2,y2,x3,y3]), priorLine]), axis=0).astype(int)
            cords.append(cordsAverage) 
            lineImage = drawRoadLines(lineImage, cordsAverage[:4], 2, 0, 255, 0)
    
    resultingFrame = cv2.addWeighted(frame, 1, lineImage, 1, 1)
    return  cords, resultingFrame

def drawRoadLines(frame, cords, thicc, r, g, b):
    cv2.line(frame, (cords[0], cords[1]), (cords[2], cords[3]), (b, g, r), thicc)
    return frame

while True:
    img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
    img_np = np.array(img)
    videoFrame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    edgeDetectedFrame = edgeDetection(videoFrame, 3, 25, 75)

    lines = cv2.HoughLinesP(edgeDetectedFrame, 2, np.pi/180, 100, np.array([]), minLineLength=15, maxLineGap=275)

    if lines is None:
        continue
    Lines = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        Lines.append((slope, intercept))

        cv2.line(videoFrame, (x1,y1), (x2,y2), (255,0,0), 10)

    LinesAverage = np.average(Lines, axis=0)
    cv2.line(videoFrame, (LinesAverage[0],LinesAverage[1]), (LinesAverage[2],LinesAverage[3]), (0,255,0), 10)
    # cords, resultingFrame = makeCoordinates(videoFrame, line, cords)

    # resultingFrame = cv2.addWeighted(resultingFrame, 1, highlightedRoad, 0.1, 1)

    cv2.imshow('videoFrame', videoFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows()