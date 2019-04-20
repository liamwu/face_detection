import cv2 as cv
import numpy as np

#change the photo to gray
def convertToGRAY(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#detect human face single frame
def detectFace(img, classifier):
    face_cascade = cv.CascadeClassifier(classifier)
    faces = face_cascade.detectMultiScale(convertToGRAY(img), scaleFactor=1.08, minNeighbors=5, minSize=(8, 8))

    for(x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv.imshow("COOL", img)

#play video with face detection
def playVideoWithFaceDetection(video, classifier):
    #read video
    video = cv.VideoCapture(video)

    while(True):
        #capture image from video
        ret, img = video.read()

        if ret == True:
            detectFace(img, classifier)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    video.release()
    cv.destroyAllWindows()


video_path = 'videos\\face_data_with_faces.mp4'
classifier = "model\\haarcascade_frontalface_alt2.xml"
playVideoWithFaceDetection(video_path, classifier)
