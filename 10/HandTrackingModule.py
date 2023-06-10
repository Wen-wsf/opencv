import cv2
import mediapipe as mp
import time
import math    # 08.增加匯入數學運算

class handDetector():
    def __init__(self, mode=False, maxHands = 2, model_comp = 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHnads = maxHands
        self.model_comp = model_comp
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        # 定義手的資訊變數
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHnads, self.model_comp, self.detectionCon, self.trackCon)
        
        # 定義畫圖單元
        self.mpDraw = mp.solutions.drawing_utils
        # 09. 定義手指頭的編號
        self.tipIds = [4, 8, 12, 16, 20] 

    def findHands(self, img, draw = False):
        # 取得multi_hand_landmarks的資料 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        # 劃出關鍵點
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  # 劃出關鍵點間的連結
        return img
    def findPosition(self, img, handNo=0, draw=True):
        # 06. 新增兩個list紀錄x與y的資訊 bbox紀錄邊界範圍與 bbox紀錄範圍
        xList = []
        yList = []
        bbox = []
        
        self.lmList = []   # 06. 將lmList修改為self.lmList

        lmList = []
        if self.results.multi_hand_landmarks:
            if (handNo == -1) :
                for hid, myHand in enumerate(self.results.multi_hand_landmarks):
                    for id, lm in enumerate(myHand.landmark):
                        #print(id, lm)  #印出檢查關鍵點座標
                        h, w, c = img.shape
                        cx, cy=int(lm.x*w), int(lm.y*h)
                        
                        # 06. 將 cx 與 cy 紀錄至 xList 與 yList
                        xList.append(cx)
                        yList.append(cy)

                        #print(id, cy, cx)  #印出轉換後的關鍵點座標
                        
                        self.lmList.append([id, cx, cy])    # 06. 將lmList修改為self.lmList

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
                        xList.append(cx)
                        yList.append(cy)
                        #print(id, cy, cx)  #印出轉換後的關鍵點座標
                        self.lmList.append([id, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        # 06. 由 xlist 與 ylist 中找到最大與最小值
        if len(xList) != 0:
            xmin, xmax = min(xList), max(xList)
        else:
            xmin, xmax = 0, -20   #設定最小位置，最大值-20，因為顯示時+20
        if len(yList) != 0:
            ymin, ymax = min(yList), max(yList)
        else:
            ymin, ymax = 0, -20   #設定最小位置，最大值-20，因為顯示時+20

        # 06. 設定手的邊界範圍
        bbox = xmin, ymin, xmax, ymax
        
        #print (bbox)
        if draw:
            cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (0, 255, 0), 3)

        return self.lmList, bbox
    # 09. 增加檢查手指是否向上的函式
    def fingersUp(self):
        fingers = []
        
        # Thumb
        if self.lmList[1][1] > self.lmList[0][1]:
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
                            
        # 4 Fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)     
            else:
                fingers.append(0)    
        
        return fingers
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)            
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        
        return length, img, [x1, y1, x2, y2, cx, cy]  #回傳長度與圖形
