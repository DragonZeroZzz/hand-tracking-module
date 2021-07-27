import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while 1:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks:
        for handLmk in result.multi_hand_landmarks:
            for id, lm in enumerate(handLmk.landmark):
               h,w,c = img.shape
               cx,cy = int(lm.x*w),int(lm.y*h)
               print(id,cx,cy)
               if id==8:
                   cv2.circle(img,(cx,cy),15,(0,255,255),cv2.FILLED)

            mpDraw.draw_landmarks(img,handLmk, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,60),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)

