import dlib
import numpy as np
import cv2
import pickle
link = "C:\\Users\\Admin\\desktop\\data\\dataHOG_LBPH\\Data.pickle"
data = pickle.loads(open(link, "rb").read())
hog_face_detector = dlib.get_frontal_face_detector()
path = "C:\\Users\\Admin\\desktop\\data\\Data\\"
cap = cv2.VideoCapture(0)
rec = cv2.face_LBPHFaceRecognizer.create()
rec.read("C:\\Users\\Admin\\desktop\\data\\dataHOG_LBPH\\trainingdata.yml")
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
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