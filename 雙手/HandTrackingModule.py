import imp
import cv2
from cv2 import split
import mediapipe as mp
import time,math
import numpy as np
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
        self.tipIds = [4, 8, 12, 16, 20, 25, 29, 33, 37,41] 

    def findHands(self, img, draw = False):
        # 取得multi_hand_landmarks的資料 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(self.results.multi_hand_landmarks)
        #print("-----------------------------------------")
        # 劃出關鍵點
        if draw:
            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
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
        #print(np.shape(self.lmList))
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
        if np.shape(self.lmList)==(42,3):
            if self.lmList[22][1] > self.lmList[21][1]:
                if self.lmList[self.tipIds[5]][1] > self.lmList[self.tipIds[5]-1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if self.lmList[self.tipIds[5]][1] < self.lmList[self.tipIds[5]-1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)   
            for id in range(6,10):
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

    def HW16(self,img,fingers,overlayList):
        print(fingers)
        if np.shape(fingers)==(5,):
            if fingers==[0,1,0,0,0]:
                h, w, c = overlayList[0].shape
                img[0:h, 0:w] = overlayList[0]
                cv2.rectangle(img, (20, 225), (170, 425), (255,0,255), cv2.FILLED)
                cv2.putText(img, str(1), (45, 375), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 15)
            elif fingers==[0,1,1,0,0]:
                h, w, c = overlayList[1].shape
                img[0:h, 0:w] = overlayList[1]
                cv2.rectangle(img, (20, 225), (170, 425), (255,0,255), cv2.FILLED)
                cv2.putText(img, str(2), (45, 375), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 15)
            elif fingers==[0,1,1,1,0]:
                h, w, c = overlayList[2].shape
                img[0:h, 0:w] = overlayList[2]
                cv2.rectangle(img, (20, 225), (170, 425), (255,0,255), cv2.FILLED)
                cv2.putText(img,str(3), (45, 375), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 15)
            elif fingers==[0,1,1,1,1]:
                h, w, c = overlayList[3].shape
                img[0:h, 0:w] = overlayList[3]
                cv2.rectangle(img, (20, 225), (170, 425), (255,0,255), cv2.FILLED)
                cv2.putText(img, str(4), (45, 375), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 15)
            elif fingers==[1,0,0,0,0]:
                h, w, c = overlayList[4].shape
                img[0:h, 0:w] = overlayList[4]
                cv2.rectangle(img, (20, 225), (170, 425), (255,0,255), cv2.FILLED)
                cv2.putText(img,str(5), (45, 375), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 15)
            elif fingers==[1,1,0,0,0]:
                h, w, c = overlayList[5].shape
                img[0:h, 0:w] = overlayList[5]
                cv2.rectangle(img, (20, 225), (170, 425), (255,0,255), cv2.FILLED)
                cv2.putText(img, str(6), (45, 375), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 15)
            elif fingers==[1,1,1,0,0]:
                h, w, c = overlayList[6].shape
                img[0:h, 0:w] = overlayList[6]
                cv2.rectangle(img, (20, 225), (170, 425), (255,0,255), cv2.FILLED)
                cv2.putText(img, str(7), (45, 375), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 15)
            elif fingers==[1,1,1,1,0]:
                h, w, c = overlayList[7].shape
                img[0:h, 0:w] = overlayList[7]
                cv2.rectangle(img, (20, 225), (170, 425), (255,0,255), cv2.FILLED)
                cv2.putText(img,str(8), (45, 375), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 15)
            elif fingers==[1,1,1,1,1]:
                h, w, c = overlayList[8].shape
                img[0:h, 0:w] = overlayList[8]
                cv2.rectangle(img, (20, 225), (170, 425), (255,0,255), cv2.FILLED)
                cv2.putText(img,str(9), (45, 375), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 15)
            elif fingers==[0,0,0,0,0]:
                h, w, c = overlayList[9].shape
                img[0:h, 0:w] = overlayList[9]
                cv2.rectangle(img, (20, 225), (170, 425), (255,0,255), cv2.FILLED)
                cv2.putText(img, str(0), (45, 375), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 15)
        elif  np.shape(fingers)==(10,):
            split_list=np.array_split(fingers,2)
            h_list=[]
            w_list=[]
            str_list=[]
            img_list=[]
            for finger in split_list:    
                if all(finger==[0,1,0,0,0]):#1
                    h, w, c = overlayList[0].shape
                    h_list.append(h)
                    w_list.append(w)
                    str_list.append(1)
                    img_list.append(0)

                elif all(finger==[0,1,1,0,0]):#2
                    h, w, c = overlayList[1].shape
                    h_list.append(h)
                    w_list.append(w)
                    str_list.append(2)
                    img_list.append(1)
                elif all(finger==[0,1,1,1,0]):#3
                    h, w, c = overlayList[2].shape
                    h_list.append(h)
                    w_list.append(w)
                    str_list.append(3)
                    img_list.append(2)
                elif all(finger==[0,1,1,1,1]):#4
                    h, w, c = overlayList[3].shape
                    h_list.append(h)
                    w_list.append(w)
                    str_list.append(4)
                    img_list.append(3)
                elif all(finger==[1,0,0,0,0]):#5
                    h, w, c = overlayList[4].shape
                    h_list.append(h)
                    w_list.append(w)
                    str_list.append(5)
                    img_list.append(4)
                elif all(finger==[1,1,0,0,0]):#6
                    h, w, c = overlayList[5].shape
                    h_list.append(h)
                    w_list.append(w)
                    str_list.append(6)
                    img_list.append(5)
                elif all(finger==[1,1,1,0,0]):#7
                    h, w, c = overlayList[6].shape
                    h_list.append(h)
                    w_list.append(w)
                    str_list.append(7)
                    img_list.append(6)
                elif all(finger==[1,1,1,1,0]):#8
                    h, w, c = overlayList[7].shape
                    h_list.append(h)
                    w_list.append(w)
                    str_list.append(8)
                    img_list.append(7)
                elif all(finger==[1,1,1,1,1]):#9
                    h, w, c = overlayList[8].shape
                    h_list.append(h)
                    w_list.append(w)
                    str_list.append(9)
                    img_list.append(8)
                elif all(finger==[0,0,0,0,0]):#0
                    h, w, c = overlayList[9].shape
                    h_list.append(h)
                    w_list.append(w)
                    str_list.append(0)
                    img_list.append(9)
            for id,h in enumerate(h_list[::-1]):
                w=w_list[id]
                img_n=img_list[id]
                if id == 0:
                    img[0:h, 0:w] = overlayList[img_n]
                else:
                    w_1=w_list[id-1]
                    img[0:h, w_1:w+w_1] = overlayList[img_n]
            out_num=0
            str_list=str_list[::-1]
            for id,num in enumerate(str_list):
                #print(out_num+num*10**id,"=",out_num,"+",num,"*10^",id)
                out_num= out_num+num*10**id
            cv2.rectangle(img, (20, 225), (300, 425), (255,0,255), cv2.FILLED)
            cv2.putText(img, str(out_num), (45, 375), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 15)    
        return img