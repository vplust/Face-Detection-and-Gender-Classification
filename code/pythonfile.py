import cv2
import dlib
import numpy
import os 
import sys
from svm import *
from svmutil import *


PREDICTOR_PATH = "/home/saksham/Desktop/landmark/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


face_detected = 0

image = sys.argv[1]

im = cv2.imread(image)
im_tuple = tuple(im.shape[1::-1])

if im_tuple[0]  > 1000 or im_tuple[1] > 750 :
    im = cv2.resize(im, (0,0), fx=0.75, fy=0.75) 
dets = detector(im, 1)
print len(dets)

for k, d in enumerate(dets):
    face_detected = face_detected + 1
    y = d.top()
    x = d.left()
    s = d.right()
    t = d.bottom()
    landmarks =  numpy.matrix([[p.x, p.y] for p in predictor(im, d).parts()])

    
    # calculating height of eye
    he =((float(landmarks[41][0,1])-float(landmarks[37][0,1])) + (float(landmarks[40][0,1])-float(landmarks[38][0,1])) +
        (float(landmarks[47][0,1])-float(landmarks[43][0,1])) + (float(landmarks[46][0,1])-float(landmarks[44][0,1])))/4

    # distance between eyes and eyebrows
    debe = ((float(landmarks[36][0,1])-float(landmarks[17][0,1])) + (float(landmarks[39][0,1])-float(landmarks[21][0,1])) +
        (float(landmarks[42][0,1])-float(landmarks[22][0,1])) + (float(landmarks[45][0,1])-float(landmarks[26][0,1])))/4

    # height of nose 
    hn = float(landmarks[33][0,1])-float(landmarks[27][0,1])

    # distance between lip and nose 
    dln = float(landmarks[51][0,1])-float(landmarks[33][0,1])

    # distance between eyes
    dee  = (float(landmarks[45][0,0])+float(landmarks[42][0,0]))/2  - (float(landmarks[39][0,0])+float(landmarks[36][0,0]))/2

    # center of right eye
    a = (float(landmarks[45][0,0])+float(landmarks[42][0,0]))/2
    b = ((float(landmarks[47][0,1])+float(landmarks[43][0,1]))/2 + (float(landmarks[46][0,1])+float(landmarks[44][0,1]))/2)/2
    cre = (a,b)
    
    # center of left eye
    c = (float(landmarks[39][0,0])+float(landmarks[36][0,0]))/2 
    d = ((float(landmarks[41][0,1])+float(landmarks[37][0,1]))/2 + (float(landmarks[40][0,1])+float(landmarks[38][0,1]))/2)/2
    cle = (c,d)

    # eye to upperlip distance
    dule = float(landmarks[51][0,1]) - ((b+d)/2)

    # eye to nose distance
    dne = float(landmarks[33][0,1]) - ((b+d)/2)

    # eye to chin distance 
    dche = float(landmarks[8][0,1] - ((b+d)/2))
    
    #eyebrow curve
    ec = float(((landmarks[17][0,1] + landmarks[21][0,1])/2 - landmarks[19][0,1]) + \
        ((landmarks[22][0,1] + landmarks[26][0,1])/2 - landmarks[24][0,1])/2)

    # distance between upperlip and nose 
    dln = float(landmarks[51][0,1]) - float(landmarks[33][0,1])

    # chin curve
    cc  = (float(landmarks[7][0,1]) + float(landmarks[9][0,1]))/2 - float(landmarks[8][0,1])

    

    ratio1 = dee/hn
    ratio2 = dee/dule
    ratio3 = dne/dche
    ratio4 = dee/dche
    ratio5 = dne/dule
    if ec ==0:
        ratio6 = dee
    else:
        ratio6 = dee/ec
    ratio7 = dee/debe
    ratio8 = dee/dln
    ratio9 = dee/he
    if cc == 0:
        ratio10 = dee
    else:
        ratio10 = dee/cc

    values = [{1:ratio1, 2:ratio2, 3:ratio3, 4:ratio4, 5:ratio5, 6:ratio6, 7:ratio7, 8:ratio8, 9:ratio9, 10:ratio10}]
    p_l = [0]*len(values)
    model = svm_load_model('/home/saksham/Desktop/minorprojectcodes/dlib2training.model')
    a,b,val = svm_predict(p_l,values,model)
    if int(a[0]) == -1:
        cv2.rectangle(im, (x,y), (s,t), (255,0,0),2)
        cv2.putText(im,'Female', (x-10,y-10), 
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.5,
                    thickness = 1,
                    color=(255, 0, 255))

    elif int(a[0]) == 1:
        cv2.rectangle(im, (x,y), (s,t), (255,0,0),2)

        cv2.putText(im,'Male', (x-10,y-10), 
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.5,
                    thickness = 1,
                    color=(255, 0, 0))


cv2.imshow('Result',im)
cv2.waitKey(0)
cv2.destroyAllWindows()




