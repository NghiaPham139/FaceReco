
import numpy as np
import cv2
import os
path = "C:\\Users\\Admin\\desktop\\data\\Data\\"
face_cascade = cv2.CascadeClassifier('C:\\Users\\Admin\\desktop\\data\\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
id = raw_input('enter user id ')
try:
    # Create target Directory
    os.mkdir(path+str(id))
    print("Directory " , path+str(id),  " Created ")
except Exception:
    print("Directory " , path+str(id) ,  " already exists")
sampleN=0;
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if faces is not None:
        for (x,y,w,h) in faces:
            sampleN=sampleN+1;
            cv2.imwrite(path+str(id)+ "\\" +str(sampleN)+ ".jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.waitKey(100)
    cv2.imshow('img',img)
    cv2.waitKey(1)
    if sampleN > 20:
        break

cap.release()
cv2.destroyAllWindows()