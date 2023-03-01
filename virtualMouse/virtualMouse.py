import time

import autopy as autopy
import cv2
import numpy as np

import handTrackingModule as htm

wCam, hCame = 640, 480
frameR = 100  # frame reduction
smoothening = 7

plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(-1)
cap.set(3, wCam)
cap.set(4, hCame)
pTime = 0
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

while True:
    # find hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # get the tip index and middle finger
    if lmList:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # only index finger : moving mode
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCame - frameR), (255, 0, 255), 2)
        if fingers[1] == 1 and fingers[2] == 0:
            # convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCame - frameR), (0, hScr))
            # smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # move mouse
            autopy.mouse.move(clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        # both index and middle fingers are up: clicking mode
        # find distance between fingers
        # click mouse if distance short
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # frame rate
    cTime = time.time()
    fps = int(1 / (cTime - pTime))
    pTime = cTime

    # display
    cv2.putText(img, str(fps), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
