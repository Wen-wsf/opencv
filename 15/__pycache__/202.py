import cv2
import time
import numpy as np
import mediapipe as mp
import math
import PoseModule as pd    # 05.匯入姿態偵測器

# 01. 取得鏡頭影像的基本架構
wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
# 02. FPS時間設定
pTime = 0
# 05.匯入姿態偵測器
detector = pd.PoseDetector()
# 08. 
count = 0
count1 = 0
dir = 0
color = (255, 0, 255)
per = 0
bar = 0

while True:
    success, img = cap.read()
    # 水平翻轉畫面
    img = cv2.flip(img, 1) 
    # 05.匯入姿態偵測器
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)
    # 07.計算角度
    if len(lmList) != 0:
        #左手臂
        angle = detector.findAngle(img, 12, 14, 16, color=(255,0,0))
        # 08. 換算百分比，右臂最小角度為30，最大角度為150
        per = np.interp(angle, (30, 150), (100, 0))
        angle1 = detector.findAngle(img, 11, 13, 15, color=(0,255,0))
        bar = np.interp(angle1, (30, 150), (100, 650))
    if len(lmList) != 0: 
        # 08. 設定顏色
        color = (91, 101, 227)
        if per == 100:
            color = (166, 145, 56)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (166, 145, 56)
            if dir == 1:
                count += 0.5
                dir = 0
    if len(lmList) != 0: 
        # 08. 設定顏色
        color = (91, 101, 227)
        if bar == 100:
            color = (166, 145, 56)
            if dir == 0:
                le += 0.5
                dir = 1
        if bar == 0:
            color = (166, 145, 56)
            if dir == 1:
                le += 0.5
                dir = 0     
    # 08. 劃出長條圖
    cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
    cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
    cv2.putText(img, f'{int(per)}%', (1080, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)
    # 08. 劃出記分板
    if count<10:
        c = 0
    elif count<100:
        c = 1
    else:
        c = 2
    cv2.rectangle(img, (0, 450), (250+c*120, 720), (92, 91, 96), cv2.FILLED)
    cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                (76, 231, 253), 25) 
        # 08. 劃出記分板
    if count1< 0:
        a = 0
    elif count1 < 100:
        a = 1
    else:
        a = 2
    cv2.rectangle(img, (1280, 720), (1020,500), (92, 91, 96), cv2.FILLED)
    cv2.putText(img, str(int(count1)), (1150, 650), cv2.FONT_HERSHEY_PLAIN, 15,
                (76, 231, 253), 25) 
    # 02. 顯示FPS (frame rate)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (wCam-120, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)

    cv2.imshow("Image", img)
    # 01. 按Q離開
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()