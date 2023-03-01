import cv2
import  glob
import time
import handTrackingModule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(-1)
cap.set(3, wCam)
cap.set(4, hCam)
imgDir = glob.glob("./finger/*")
overlayList = []
pTime = 0

detector = htm.handDetector()
tipIds = [4, 8, 12, 16, 20]
totalFingers = 0

for im in imgDir:
    image = cv2.imread(im)
    overlayList.append(image)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    if lmList:
        fingers = []
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for idx in range(1, 5):
            if lmList[tipIds[idx]][2] < lmList[tipIds[idx]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)
        imgDic = {0: 5, 1:2, 2:4, 3:0, 4:3, 5:1}

        h, w, c = overlayList[imgDic[totalFingers]].shape
        img[0:h, 0:w] = overlayList[imgDic[totalFingers]]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime = cTime

    cv2.putText(img, f"FPS: {fps}", (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

