import cv2
import mediapipe as mp
import time



class HandDetector():
    def __init__(self, mode=False, maxHands= 2, detectionConf = 0.5, trackConf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
                                        max_num_hands = self.maxHands,
                                        static_image_mode = self.mode,
                                        min_detection_confidence = detectionConf,
                                        min_tracking_confidence = trackConf, 
                                        )
        #self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4,8,12,16,20]

    def findHands(self, img, draw=True):

        #Convert img to RGB format for hands mp object
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        #get hand landmarks
        self.handsLd = results.multi_hand_landmarks
        
        #Draw single hand in frame if hand exist
        if self.handsLd:
            for singHandLms in self.handsLd:
                if draw:
                 self.mpDraw.draw_landmarks(img,singHandLms,self.mpHands.HAND_CONNECTIONS)
                
        return img        

    def findPosition(self, img, handNo = 0, draw = True):

        self.lmList = []

        #Get the landmarks information
        #id is id of landmark in hand
        #lm is a object with ratio postion of landmark
        if self.handsLd:
            #get choise hand
            myHand = self.handsLd[handNo]
            for id, lm in enumerate(myHand.landmark):
                #convert ration position in pixels
                h, w, c = img.shape
                #get center for all landmarks
                cx,cy = int(lm.x*w), int(lm.y*h)
                self.lmList .append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
        return self.lmList 

    def fingersUp(self):
        fingers = []

        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():

    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    handDetector = HandDetector()
    while True:
        succes, img = cap.read()
        img = handDetector.findHands(img)
        lmList = handDetector.findPosition(img)
        # if lmList:
        #     print(lmList[1])
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)



if __name__ == '__main__':
    main()