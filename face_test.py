import cv2 as cv
import argparse
import matplotlib.pyplot as plt
import time

#change the photo to gray
def convertToGRAY(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#show photo, escape by press any key
def showImg(wname, img):
    cv.imshow(wname, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def detectFaces(face_path, calssifier):
    test = cv.imread(face_path)
    test_gray = convertToGRAY(test)

    #set cascade classifier
    face_cascade = cv.CascadeClassifier(calssifier)

    #face detection function
    faces = face_cascade.detectMultiScale(test_gray, scaleFactor=1.08, minNeighbors=5, minSize=(8, 8))
    print(len(faces))

    #locate the face in original photo
    for(x, y, w, h) in faces:
        cv.rectangle(test, (x, y), (x + w, y + h), (0, 255, 0), 1)

    showImg("out", test)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagepath", required=True, help="set image path")
ap.add_argument("-m", "--classifiermodel", required=True, help="set face calssifier model")
args = vars(ap.parse_args())

#face_path = "faces\\faces.jpg"
#classifier = "model\\haarcascade_frontalface_alt2.xml"
detectFaces(args["imagepath"], args["classifiermodel"])