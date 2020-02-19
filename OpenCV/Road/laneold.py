import cv2
import numpy as np

leftLine = [0,0,0,0,0]
rightLine = [0,0,0,0,0]

def drawCarDir(frame):
    lineImage = np.zeros_like(frame)
    cv2.line(lineImage, (int(frame.shape[1]/2),frame.shape[0]), (int(frame.shape[1]/2),int(frame.shape[0]/1.3)), (0,0,255), 10)
    return lineImage

def makeCoordinates(frame, lineParameters, priorLine):
    if not np.isnan(lineParameters).any():
        slope, intercept = lineParameters
        y1 = frame.shape[0]
        y2 = int(y1*(3/5))
        x1 = int((y1-intercept)/slope)
        x2 = int((y2-intercept)/slope)
        cords = [x1,y1,x2,y2]
        cords.append(0)
        return cords
    priorLine[4] = 256
    return priorLine

def averageByK(frame, lines):
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
    # ge tillbaks de två åvan
    global leftLine
    global rightLine
    leftLine = makeCoordinates(frame, leftLinesAverage, leftLine)
    rightLine = makeCoordinates(frame, rightLinesAverage, rightLine)
    return np.array([leftLine, rightLine])

def drawRoadLines(frame, lines, thicc):
    lineImage = np.zeros_like(frame)
    for x1, y1, x2, y2, new in lines:
        cv2.line(lineImage, (x1, y1), (x2, y2), (0, 255, int(new)), thicc)
    return lineImage

def highlightRoad(frame, lines):
    roadImage = np.zeros_like(frame)
    # cords = lines
    # del lines[0][-1]
    # del lines[1][-1]
    # remove index -1 and index 4
    cv2.fillPoly(roadImage, lines, color=(0, 255, 0))
    return roadImage

def edgeDetection(frame, blurSize, lowT, highT):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurredFrame = cv2.GaussianBlur(grayFrame, (blurSize, blurSize), 0)
    edgeDetectedFrame = cv2.Canny(blurredFrame, lowT, highT)
    return edgeDetectedFrame

def regionOfInterest(frame, frameHeight, xLeft, xRight, xMiddle, yMiddle):
    FOV = np.array([[(xLeft, frameHeight), (xRight, frameHeight), (xMiddle, yMiddle)]])
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, FOV, 255)
    reducedFOVFrame = cv2.bitwise_and(frame, mask)
    return reducedFOVFrame

def processingSteps(frame):
    edgeDetectedFrame = edgeDetection(frame, 3, 25, 75)
    reducedFOVFrame = regionOfInterest(edgeDetectedFrame, frame.shape[0], 200, 1100, 560, 260)
    lines = cv2.HoughLinesP(reducedFOVFrame, 2, np.pi/180, 100, np.array([]), minLineLength=15, maxLineGap=275)
    linesAverage = averageByK(frame, lines)
    # boxFrame = highlightRoad(frame, linesAverage)
    lineFrame = drawRoadLines(frame, linesAverage, 8)
    resultingFrame = cv2.addWeighted(frame, 0.85, lineFrame, 1, 1)



    # gör cords här och använd i drawroadlines och highlight road

    resultingFrame = cv2.addWeighted(resultingFrame, 0.85, drawCarDir(frame), 1, 1)
    # resultingFrame = cv2.addWeighted(resultingFrame, 0.85, boxFrame, 1, 1)

    # skriv vinkeln på bilen jämfört med vägen
    # ha två areor som linjerna måste gå igenom
    # distancen mellan mittpunkten och linjerna

    return reducedFOVFrame, resultingFrame

capture = cv2.VideoCapture('ontheroadagain.mp4')
    
while capture.isOpened():
    _, videoFrame = capture.read()
    reducedFOVFrame, resultingFrame = processingSteps(videoFrame)
    cv2.imshow('edges', reducedFOVFrame)
    cv2.imshow('result', resultingFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

capture.release()
cv2.destroyAllWindows()