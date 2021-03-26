
 
# Real-Time Face Mask Detection with Python
# Machine Learning Project on Real-Time Face Mask Detection with Python

import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense

cnn=Sequential([Conv2D(filters=100,kernel_size=(3,3),
                       activation='relu'),
                MaxPooling2D(pool_size=(2,2)),
                Conv2D(filters=100,kernel_size=(3,3),
                       activation='relu'),
                MaxPooling2D(pool_size=(2,2)),
                Flatten(),
                Dropout(0.5),
                Dense(50),
                Dense(35),
                Dense(2)])
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

labels_dict={0:'No mask',1:'Mask'}
color_dict={0:(0,0,255),1:(0,255,0)}
imgsize=4
camera=cv2.VideoCapture(0)

classifier=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
    rval,im=camera.read()
    im=cv2.flip(im,1,1)
    imgs=cv2.resize(im,(im.shape[1] // imgsize,im.shape[0] // imgsize))
    face_rec=classifier.detectMultiScale(imgs)

    for i in face_rec:
        (x,y,l,w)=[v*imgsize for v in i]
        face_img=im[y:y+w,x:x+l]
        resized=cv2.resize(face_img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped=np.vstack([reshaped])
        result=cnn.predict(reshaped)
        label=np.argmax(result,axis=1)[0]
        cv2.rectangle(im,(x,y),(x+l,y+w),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+1,y),color_dict[label],-1)
        cv2.putText(im,labels_dict[label],(x,y-10),
                    cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)


    cv2.imshow('Live',im)
    key=cv2.waitKey(10)
    if key==27:   # The Esc key
        break

camera.release()
cv2.destroyAllWindows()


