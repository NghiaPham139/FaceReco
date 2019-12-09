import os
import numpy as np
import cv2
from PIL import Image
import imutils.paths as paths
import pickle
# For face recognition we will the the LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "C:\\Users\\Admin\\desktop\\data\\Data\\"
face_cascade = cv2.CascadeClassifier('C:\\Users\\Admin\\desktop\\data\\haarcascade_frontalface_alt.xml')
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
        faces=face_cascade.detectMultiScale(gray)
        for (x, y, w, h) in faces:
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
recognizer.save("C:\\Users\\Admin\\desktop\\data\\trainingdata.yml")
data={"IDs":Ids,"Names":names}
output = open("C:\\Users\\Admin\\desktop\\data\\Data.pickle", "wb")
pickle.dump(data, output)
output.close()
cv2.destroyAllWindows()
print (Ids)
print (names)