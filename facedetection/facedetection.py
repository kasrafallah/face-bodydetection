import numpy as np
import mediapipe as mp
import cv2
cap = cv2.VideoCapture(0)
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
FaceDetection = mpFaceDetection.FaceDetection()
mpFaceMesh = mp.solutions.face_mesh
FaceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1)
while True:

    success, img = cap.read()
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = FaceMesh.process(imageRGB)
    if results.multi_face_landmarks:
        for detections in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,detections,mpFaceMesh.FACE_CONNECTIONS,drawSpec,drawSpec)


    cv2.imshow('Image',img)
    cv
