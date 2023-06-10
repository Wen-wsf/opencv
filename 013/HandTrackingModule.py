import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands = 2, model_comp = 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_comp = model_comp
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # 02. 定義手的資訊變數
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.model_comp,self.detectionCon,self.trackCon)
        # 03. 定義畫圖單元
        self.mpDraw =mp.solutions.drawing_utils


    def findHands(self, img, draw = True):
        imgRGB=cv2.cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
    
        # 03. 劃出關鍵點
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
    
        return img 
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            if (handNo == -1) :
                for hid, myHand in enumerate(self.results.multi_hand_landmarks):
                    for id, lm in enumerate(myHand.landmark):
                        #print(id, lm)  #印出檢查關鍵點座標
                        h, w, c = img.shape
                        cx, cy=int(lm.x*w), int(lm.y*h)
                        #print(id, cy, cx)  #印出轉換後的關鍵點座標
                        lmList.append([id, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 5, (hid*100+100, 255-hid*100, 255), cv2.FILLED)
            else:
                if len(self.results.multi_hand_landmarks) > handNo:
                    myHand = self.results.multi_hand_landmarks[handNo]
                    for id, lm in enumerate(myHand.landmark):
                        #print(id, lm)  #印出檢查關鍵點座標
                        h, w, c = img.shape
                        cx, cy=int(lm.x*w), int(lm.y*h)
                        #print(id, cy, cx)  #印出轉換後的關鍵點座標
                        lmList.append([id, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmList
