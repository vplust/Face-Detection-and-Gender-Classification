import dlib
import numpy
import os 
import cv2
import sys
import glob2
from skimage import io


PREDICTOR_PATH = "/home/saksham/Desktop/landmark/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='/home/saksham/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

path = 'datasetnew/*/**/*.jpg'

face_detect_count = 0
total_faces = len(glob2.glob(path))

detector = dlib.get_frontal_face_detector()

for f in glob2.glob(path):
    gender = f.split('/')[-3] 
    faces= f.split('/')[-2]
    im = cv2.imread(f)
    dets = detector(im, 1)
    faces = cascade.detectMultiScale(im, 1.3, 5)

    for k, d in enumerate(dets):
        
        #roi_img = im[y:y+h, x:x+w]
        #rect=dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
        landmarks =  numpy.matrix([[p.x, p.y] for p in predictor(im, d).parts()])

        face_detect_count   = face_detect_count+1
        
        # calculating height of eye
        he =((float(landmarks[41][0,1])-float(landmarks[37][0,1])) + (float(landmarks[40][0,1])-float(landmarks[38][0,1])) +
            (float(landmarks[47][0,1])-float(landmarks[43][0,1])) + (float(landmarks[46][0,1])-float(landmarks[44][0,1])))/4

        # distance between eyes and eyebrows
        debe = ((float(landmarks[36][0,1])-float(landmarks[21][0,1])) + (float(landmarks[39][0,1])-float(landmarks[21][0,1])) +
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

        ec = float(((landmarks[17][0,1] + landmarks[21][0,1])/2 - landmarks[19][0,1]) + \
            ((landmarks[22][0,1] + landmarks[26][0,1])/2 - landmarks[24][0,1])/2)

        dln = float(landmarks[51][0,1]) - float(landmarks[33][0,1])

        cc  = (float(landmarks[7][0,1]) + float(landmarks[9][0,1]))/2 - float(landmarks[8][0,1])



        de  =((float(landmarks[36][0,1])-float(landmarks[17][0,1])) + (float(landmarks[39][0,1])-float(landmarks[21][0,1])) +
            (float(landmarks[42][0,1])-float(landmarks[22][0,1])) + (float(landmarks[45][0,1])-float(landmarks[26][0,1])))/4


        ratio1 = dee/hn
        print 'ratio1'
        print ratio1 
        ratio2 = dee/dule
        print 'ratio2'
        print ratio2
        ratio3 = dne/dche
        print 'ratio3'
        print ratio3
        ratio4 = dee/dche
        print 'ratio4'
        print ratio4
        ratio5 = dne/dule
        print 'ratio5'
        print ratio5
        if ec ==0:
            ratio6 = dee
        else:
            ratio6 = dee/ec
        print 'ratio6'
        print ratio6
        ratio7 = dee/de
        print 'ratio7'
        print ratio7
        ratio8 = dee/dln
        print 'ratio8'
        print ratio8
        ratio9 = dee/he
        print 'ratio9'
        print ratio9
        if cc == 0:
            ratio10 = dee
        else:
            ratio10 = dee/cc
        print 'ratio10'
        print ratio10

        print gender
        # generating text files
        if  gender == 'females':
            with open("Outputdlibnew.txt", "a") as text_file:
                text_file.write("-1 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} 9:{} 10:{}\n" .format(ratio1, ratio2, ratio3, ratio4, ratio5,
                    ratio6, ratio7, ratio8, ratio9, ratio10))

        else:
            with open("Outputdlibnew.txt", "a") as text_file:
                text_file.write("1 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} 9:{} 10:{}\n" .format(ratio1, ratio2, ratio3, ratio4, ratio5,
                    ratio6, ratio7, ratio8, ratio9, ratio10))
face_success_ratio = (float(face_detect_count)/float(total_faces))*100

print 'success ratio is :{}'.format(face_success_ratio)
