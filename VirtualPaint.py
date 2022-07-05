import cv2
import numpy as np
import time
import os
import Modules.HandTrackingModule as htm



#read header images
folderPath = './Header'
listHeaderImgs = os.listdir(folderPath)
overlayList = [cv2.imread(os.path.join(folderPath,imgPath)) for imgPath in listHeaderImgs]
header = overlayList[0]
drawColor = (255,0,255)
wHeader, hHeader, cHeader = header.shape

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
pTime = 0
cTime = 0

handDetector = htm.HandDetector(detectionConf = 0.85)
xp,yp = 0,0
brushThickness = 15
eraserThickness = 60
imgCanvas = np.zeros((720, 1280, 3), np.uint8)#h, w, c

while True:
    #import image
    succes, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = handDetector.findHands(img)
    lmList = handDetector.findPosition(img, draw=False)
    if len(lmList) !=0:
        
        #get tip finger id
        x1,y1 = lmList[8][1:]
        #get middle finger id
        x2,y2 = lmList[12][1:]

        # 3. Check wich fingers are up 
        fingers=handDetector.fingersUp()
        #print(fingers)
        # 4. If Selection mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp,yp = 0,0
            if y1 < 125:
                if 200 < x1 < 370:
                    header = overlayList[0]
                    drawColor = (255,0,255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif 820 < x1 < 1020:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                elif x1 > 1020:
                    header = overlayList[3]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1,y1-25), (x2,y2+25), drawColor, cv2.FILLED)
        # 5. If Drawing Mode
        if fingers[1] and fingers[2] == False:
            if xp == 0 and yp ==0:
                xp,yp = x1,y1

            #Especial condition for Eraser, increase thickness
            if drawColor == (0,0,0):
                cv2.circle(img, (x1,y1), eraserThickness, drawColor, cv2.FILLED)
                cv2.line(img,(xp,yp), (x1,y1), drawColor, eraserThickness)
                cv2.line(imgCanvas,(xp,yp), (x1,y1), drawColor, eraserThickness)
            else:
                cv2.circle(img, (x1,y1), brushThickness, drawColor, cv2.FILLED)
                cv2.line(img,(xp,yp), (x1,y1), drawColor, brushThickness)
                cv2.line(imgCanvas,(xp,yp), (x1,y1), drawColor, brushThickness)
            xp,yp = x1,y1


    #Create Paint Mask
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)



    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (1200,200), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    img[:wHeader, 0:hHeader] = header
    cv2.imshow('Image', img)
    #cv2.imshow('Canvas', imgCanvas)
    cv2.waitKey(1)