import cv2 as cv
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

def detectFaces(face_path):
    test = cv.imread(face_path)
    test_gray = convertToGRAY(test)

    #set cascade classifier
    face_cascade = cv.CascadeClassifier("model\\haarcascade_frontalface_alt2.xml")

    #face detection function
    faces = face_cascade.detectMultiScale(test_gray, scaleFactor=1.1, minNeighbors=5, minSize=(8, 8))
    print(len(faces))

    #locate the face in original photo
    for(x, y, w, h) in faces:
        cv.rectangle(test, (x, y), (x + w, y + h), (0, 255, 0), 2)

    showImg("out", test)

face_path = "faces\\faces.jpg"
detectFaces(face_path)