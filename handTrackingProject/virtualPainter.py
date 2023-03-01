import cv2
import time
import glob
import numpy as np
import handTrackingModule as htm

hList = glob.glob("./header/*.png")
headerList = sorted(hList, key=lambda x: int(x.split('/')[-1][0]))
overlayList = []

for imPth in headerList:
    # print(imPth)
    image = cv2.imread(imPth)
    image = cv2.resize(image, (1280, 160))
    overlayList.append(image)

header = overlayList[0]
drawColor = (0, 0, 255)
brushThickness = 10
eraserThickness = 30


cap = cv2.VideoCapture(-1)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector()
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # find hand landmark
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        # tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # check if selection mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # print("selection mode")
            if y1 < 124:
                if 250< x1 <460:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (0, 255, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (255, 0, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
                    brushThickness = 20
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # check if draw mode:
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            # print("draw mode")
            if xp==0 and yp==0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # setting header image
    img[:160, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)

