import cv2 as cv
import numpy as np

#change the photo to gray
def convertToGRAY(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#detect human face single frame
def detectFace(img, classifier, count):
    #set the face cascade classifier
    face_cascade = cv.CascadeClassifier(classifier)
    #detect faces from the image
    faces = face_cascade.detectMultiScale(convertToGRAY(img), scaleFactor=1.2, minNeighbors=6, minSize=(48, 48))

    for(x, y, w, h) in faces:
        #sign the detected face in video
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #save the detected face
        crop_img = img[y:y+h, x:x+w]
        cv.imwrite(str(count) + ".jpg", crop_img)

    cv.imshow("COOL", img)

#play video with face detection
def playVideoWithFaceDetection(video, classifier):
    #read video
    video = cv.VideoCapture(video)

    count = 0
    while(True):
        #capture image from video
        ret, img = video.read()

        if ret == True:
            detectFace(img, classifier, count)
            count = count + 1
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    video.release()
    cv.destroyAllWindows()


video_path = 'videos\\video_test.mp4'
classifier = "model\\haarcascade_frontalface_alt2.xml"
playVideoWithFaceDetection(video_path, classifier)
