import keras
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import img_to_array
import numpy as np
import pathlib
import cv2

clf = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

model_classifier = load_model("ExpressionModel.keras")
emotion_labels = ['Happy','Angry','Sad','Neutral','Surprise','Fear']
camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    for x, y, width, height in faces:
        cv2.rectangle(frame, (x, y, x + width, y + height), (255, 255, 0), 2)
        roi_gray = gray[y:y+height, x:x+width]
        roi_gray = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)
            
            preds = model_classifier.predict((roi))[0]
            label=emotion_labels[preds.argmax()]
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        cv2.imshow("Faces", frame)

    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
