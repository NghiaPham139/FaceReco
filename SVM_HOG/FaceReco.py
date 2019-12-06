import imutils
import pickle
import cv2
import face_recognition
from sklearn import svm
import numpy as np

link = "C:\\Users\\Admin\\desktop\\data\\encodingSVM.pickle"
data = pickle.loads(open(link, "rb").read())

clf = svm.SVC(kernel = 'linear')
clf.fit(data["encodings"],data["names"])

cap = cv2.VideoCapture(0)

if cap.isOpened:
    ret, frame = cap.read()
else:
    ret = False
while (ret):
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = clf.predict(encodings)
    print (encodings)
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, str(name), (top, right + left), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0))
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
