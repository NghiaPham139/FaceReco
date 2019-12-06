
import numpy as np
import cv2
import pickle
link = "C:\\Users\\Admin\\desktop\\data\\Data.pickle"
data = pickle.loads(open(link, "rb").read())
path = "C:\\Users\\Admin\\desktop\\data\\Data\\"
face_cascade = cv2.CascadeClassifier('C:\\Users\\Admin\\desktop\\data\\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face_LBPHFaceRecognizer.create()
cv2.face.LBPHFaceRecognizer_create()
rec.read("C:\\Users\\Admin\\desktop\\data\\trainingdata.yml")
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(conf<30):
            name="unknown"
        else:
            vtri=data["IDs"].index(id)
            name=data["Names"][vtri]
        cv2.putText(img,str(name),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0))
    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()