import cv2
import mediapipe as mp
import time

class hand_detector():
    def __init__(self,mode=False,max_hand=2,detection_confidence=0.65,tracking_confidence=0.65):
        self.mode = mode
        self.max_hand = max_hand
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
       
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.max_hand,self.detection_confidence,self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hand(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handLmk in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLmk, self.mpHands.HAND_CONNECTIONS)
        return img
           
    def find_pos(self,img,myID=0,hand_No=0,draw=True):

        lmlist = []

        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[hand_No]
            for id, lm in enumerate(myhand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                # print(id,cx,cy)
                lmlist.append([id,cx,cy])
                if draw:
                    if(id == myID):
                        cv2.circle(img,(cx,cy),15,(0,255,255),cv2.FILLED)
        return lmlist

def main():
    
    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0
    detector = hand_detector()
    while 1:
        success,img = cap.read()  
        img = detector.find_hand(img)
        lmlist = detector.find_pos(img,4)
        if len(lmlist)!=0:
            print(lmlist[4])
            
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,"FPS:"+str(int(fps)),(550,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)

        cv2.imshow("Image",img)
        cv2.waitKey(1)
  
if __name__ =="__main__":
    main()