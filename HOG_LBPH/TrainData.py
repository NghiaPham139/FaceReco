import cv2
import dlib
import os
import numpy as np
from PIL import Image
import imutils.paths as paths
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
hog_face_detector = dlib.get_frontal_face_detector()
path = "C:\\Users\\Admin\\desktop\\data\\Data\\"
def getImagesWithID(path):
    imagepaths = list(paths.list_images(path))
    # print image_path
    faceSamples = []
    IDs = []
    Names=[]
    ID=0
    for (i, imagePath) in enumerate(imagepaths):
        # Read the image and convert to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        gray = np.array(PIL_img,'uint8')
        # Get the label of the image
        Name= imagePath.split(os.path.sep)[-2]
        faces = hog_face_detector(gray)
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            # Detect the face in the image
            faceSamples.append(gray[y: y + h, x: x + w])
            IDs.append(ID)
            Names.append(Name)
            ID=ID+1
        cv2.imshow("Adding faces for traning",gray)
        cv2.waitKey(10)
    return IDs, faceSamples,Names
Ids,faces,names  = getImagesWithID(path)
recognizer.train(faces, np.array(Ids))
recognizer.save("C:\\Users\\Admin\\desktop\\data\\dataHOG_LBPH\\trainingdata.yml")
data={"IDs":Ids,"Names":names}
output = open("C:\\Users\\Admin\\desktop\\data\\dataHOG_LBPH\\Data.pickle", "wb")
pickle.dump(data, output)
output.close()
cv2.destroyAllWindows()
print (Ids)
print (names)