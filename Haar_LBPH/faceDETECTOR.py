
import numpy as np
import cv2
import pickle
link = "C:\\Users\\Admin\\desktop\\data\\Data.pickle"
data = pickle.loads(open(link, "rb").read())
path = "C:\\Users\\Admin\\desktop\\data\\Data\\"
face_cascade = cv2.CascadeClassifier('C:\\Users\\Admin\\desktop\\data\\haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face_LBPHFaceRecognizer.create()
rec.read("C:\\Users\\Admin\\desktop\\data\\trainingdata.yml")
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        print (conf)
        if(conf>60):
            name="unknown"
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, str(name) + str(conf), (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255))
        else:
            vtri=data["IDs"].index(id)
            name=data["Names"][vtri]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img,str(name)+str(conf),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0))
    cv2.imshow('img',img)
    #cv2.imshow('gray',gray)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()