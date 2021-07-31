import numpy as np
import mediapipe as mp
import cv2
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
while True:
    success, img = cap.read()
    imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)

    cv2.imshow('Image',img)
    cv2.waitKey(1)
