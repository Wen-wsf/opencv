import cv2
import time
import os,numpy as np
import HandTrackingModule as htm    # 07. 導入自訂的HandTrackingModule.py

# 01. 取得鏡頭影像的基本架構
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# 02. 設定手指圖片路徑
folderPath = "D:\\python\\cv2\\finger\\Fingerimages2"
myList = os.listdir(folderPath)
print(myList)

# 03. 載入圖片至overlayList中
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    print(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

# 05. FPS時間設定
pTime = 0

# 07. 使用 HandTrackingModule.py函式擷取手部資訊
detector = htm.handDetector(detectionCon=0.75)


while True:
    success, img = cap.read()

    # 04. 水平翻轉畫面
    #img = cv2.flip(img, 1) 

    # 07. 偵測手並傳回座標位置
    img = detector.findHands(img, draw = False)
    lmList, bbox = detector.findPosition(img,handNo=-1 ,draw=True)
    #print(np.shape(lmList))
    #print(lmList)
    # 10. 偵測手並傳回座標位置
    if len(lmList) != 0: 
        # 10. 檢查那些手指向上
        fingers = detector.fingersUp()
        #print(fingers)
        img=detector.HW16(img,fingers,overlayList)
        # 10. 計算向上手指數量
        totalFingers = fingers.count(1)
        # 10. 輸出數字
        #cv2.rectangle(img, (20, 225), (170, 425), (255,0,255), cv2.FILLED)
        #cv2.putText(img, f'{int(totalFingers)}', (45, 375), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 15)
        # 11. 修改原4.載入圖片至影像中的程式並放入if len(lmList) != 0:中
        #h, w, c = overlayList[totalFingers-1].shape
        #img[0:h, 0:w] = overlayList[totalFingers-1]
    # 04. 載入圖片至影像中
    #h, w, c = overlayList[0].shape
    #img[0:h, 0:w] = overlayList[0]
    # 05. 顯示FPS (frame rate)
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