import cv2
import numpy as np
from PIL import ImageGrab

def edgeDetection(frame, blurSize, lowT, highT):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurredFrame = cv2.GaussianBlur(grayFrame, (blurSize, blurSize), 0)
    edgeDetectedFrame = cv2.Canny(blurredFrame, lowT, highT)
    return edgeDetectedFrame

def regionOfInterest(frame, cord1, cord2, cord3, cord4):
    FOV = np.array([[cord1, cord2, cord3, cord4]])
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, FOV, 255)
    reducedFOVFrame = cv2.bitwise_and(frame, mask)
    return reducedFOVFrame

def makeCoordinates(frame, lines, priorCords):
    lineImage = np.zeros_like(frame)
    cords = []
    for line, priorLine in zip(lines, priorCords):
        if np.isnan(line).any():
            cords.append(priorLine)
            lineImage = drawRoadLines(lineImage, priorLine, 2, 255, 255, 0)
        else:
            slope, intercept = line
            y1 = frame.shape[0]
            y2 = int(y1*(3.25/5))
            x1 = int((y1-intercept)/slope)
            x2 = int((y2-intercept)/slope)
            y3 = int(y1*(4/5))
            x3 = int((y3-intercept)/slope)
            cords.append(np.array([x1,y1,x2,y2,x3,y3])) 
            lineImage = drawRoadLines(lineImage, np.array([x1,y1,x2,y2]), 2, 0, 255, 0)
    resultingFrame = cv2.addWeighted(frame, 1, lineImage, 1, 1)
    return cords, resultingFrame

def averageByK(lines):
    leftLines = []
    rightLines = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            leftLines.append((slope, intercept))
        else:
            rightLines.append((slope, intercept))
    leftLinesAverage = np.average(leftLines, axis=0)
    rightLinesAverage = np.average(rightLines, axis=0)
    return np.array([leftLinesAverage, rightLinesAverage])

def drawRoadLines(frame, cords, thicc, r, g, b):
    cv2.line(frame, (cords[0], cords[1]), (cords[2], cords[3]), (b, g, r), thicc)
    return frame

def drawLines(frame, cords, thicc, r, g, b):
    lineImage = np.zeros_like(frame)
    cv2.line(lineImage,  (cords[0], cords[1]), (cords[2], cords[3]), (b, g, r), thicc)
    return lineImage

def highlightRoad(frame, lines):
    cords = np.array( [ [lines[0][0],lines[0][1]], [lines[0][2],lines[0][3]], [lines[1][2],lines[1][3]], [lines[1][0], lines[1][1]] ] )
    roadImage = np.zeros_like(frame)
    cv2.fillPoly(roadImage, pts = [cords], color=(255, 255, 0))
    return roadImage

cord = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cord.append((x,y))

while True:
    img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
    img_np = np.array(img)
    videoFrame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    cv2.imshow('frame', videoFrame)
    cv2.setMouseCallback("frame", click_event)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(cord) >= 4:
        cv2.destroyAllWindows()
        break

if len(cord) != 4:
    cord = [(1025, 650),(900, 650), (300,950), (1600,950)]

cords = [np.array([0,0,1,1,2,2]), np.array([0,0,1,1,2,2])]

while True:
    try:
        img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
        img_np = np.array(img)
        videoFrame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        edgeDetectedFrame = edgeDetection(videoFrame, 3, 25, 75)
        reducedFOVFrame = regionOfInterest(edgeDetectedFrame, cord[0], cord[1], cord[2], cord[3])
    
        lines = cv2.HoughLinesP(reducedFOVFrame, 2, np.pi/180, 100, np.array([]), minLineLength=15, maxLineGap=275)

        line = averageByK(lines)
        cords, resultingFrame = makeCoordinates(videoFrame, line, cords)

        highlightedRoad = highlightRoad(videoFrame, cords)
        resultingFrame = cv2.addWeighted(resultingFrame, 1, highlightedRoad, 0.1, 1)

        highlightedCarPOV = drawLines(resultingFrame, [int(videoFrame.shape[1]/2), int(videoFrame.shape[0]), int(videoFrame.shape[1]/2), int(videoFrame.shape[0]*4/5)], 2, 255, 255, 0)    
        resultingFrame = cv2.addWeighted(resultingFrame, 1, highlightedCarPOV, 1, 1)

        lineImage = np.zeros_like(videoFrame)
        for x1, y1, x2, y2, x3, y3 in cords:
            cv2.line(lineImage, (x3, y3+20), (x3, y3-20), (0, 255, 0), 2)
        resultingFrame = cv2.addWeighted(resultingFrame, 1, lineImage, 1, 1)

        lineImage = np.zeros_like(videoFrame)
        cv2.line(lineImage, (int((cords[0][4]+cords[1][4])/2), cords[0][5]+20), (int((cords[0][4]+cords[1][4])/2), cords[0][5]-20), (0, 255, 0), 2)
        resultingFrame = cv2.addWeighted(resultingFrame, 1, lineImage, 1, 1)

        lineImage = np.zeros_like(videoFrame)
        cv2.line(lineImage, (int((cords[0][4]+cords[1][4])/2), cords[0][5]), (int(videoFrame.shape[1]/2), cords[0][5]), (255, 255, 255), 2)
        resultingFrame = cv2.addWeighted(resultingFrame, 1, lineImage, 1, 1)

        font = cv2.FONT_ITALIC
        if int((cords[0][4]+cords[1][4])/2) - int(videoFrame.shape[1]/2) < -15:
            textSize = cv2.getTextSize("Turn Left", font, 1, 2)[0][0]
            resultingFrame = cv2.putText(resultingFrame, "Turn Left", (int(videoFrame.shape[1]/2)-int(textSize/2), videoFrame.shape[0]-100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif int((cords[0][4]+cords[1][4])/2) - int(videoFrame.shape[1]/2) > 15:
            textSize = cv2.getTextSize("Turn Right", font, 1, 2)[0][0]
            resultingFrame = cv2.putText(resultingFrame, "Turn Right", (int(videoFrame.shape[1]/2)-int(textSize/2), videoFrame.shape[0]-100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('video', cv2.pyrDown(videoFrame))
        cv2.imshow('resulting', cv2.pyrDown(resultingFrame))
        cv2.imshow('edgeDetected', cv2.pyrDown(edgeDetectedFrame))
        cv2.imshow('reducedFOV', cv2.pyrDown(reducedFOVFrame))

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    except:
        continue

cv2.destroyAllWindows()