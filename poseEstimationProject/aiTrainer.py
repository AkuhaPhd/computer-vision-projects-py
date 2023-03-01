import cv2
import time
import numpy as np
import poseModule as pm

cap = cv2.VideoCapture("pexels.mp4")
detector = pm.poseDetector()
per = 0
curDir = 0
count = 0
pTime = 0

while True:
    # img = cv2.imread("fitness.jpg")
    success, img = cap.read()
    img = cv2.resize(img, (720, 1280))
    detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if lmList:
        # left arm
        # max = 155, min = 25
        angle = detector.findAngle(img, 12, 14, 16)
        per = np.interp(angle, (25, 155), (0, 100))
        # print(int(angle), per)

        # check for the dumbbell curls
        if per == 100:
            if curDir == 0:
                count += 0.5
                curDir = 1
        if per == 0:
            if curDir == 1:
                count += 0.5
                curDir = 0

        # draw bar
        # cv2.rectangle(img, (1100, 100), (1175, 650), (0, 255, 0), 3)
        # cv2.rectangle(img, (1100, int(bar)), (1175, 650), (0, 255, 0), cv2.FILLED)
        # cv2.putText(img, f"{int(bar)}", (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

        # curl count
        cv2.rectangle(img, (500, 150), (660, 20), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (555, 110), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 10)

        # right arm
        # detector.findAngle(img, 11, 13, 15)
    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime = cTime
    cv2.putText(img, str(fps), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
