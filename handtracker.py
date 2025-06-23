import mediapipe as mp
import cv2
from cvzone.HandTrackingModule import HandDetector

#webcam
cap=cv2.VideoCapture(0)
mphands=mp.solutions.hands
hands=mphands.Hands()
Draw=mp.solutions.drawing_utils

while True:
    # Read the image from the webcam
    success,img =cap.read()
    img=cv2.flip(img,1)  # Flip the image horizontally

    # Detection of hands
    # Using mediapipe hands
    imageRgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    res=hands.process(imageRgb)

    if res.multi_hand_landmarks:
        for hand_land in res.multi_hand_landmarks:
            for id,lm in enumerate(hand_land.landmark):
                # id is the index of the landmark
                h,w,c=img.shape
                xx,yy=int(lm.x*w),int(lm.y*h)
                print(id,xx,yy)

            Draw.draw_landmarks(img, hand_land, mphands.HAND_CONNECTIONS)

    cv2.imshow("Image",img)
    cv2.waitKey(1)