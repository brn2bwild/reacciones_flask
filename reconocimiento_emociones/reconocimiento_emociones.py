import cv2
import os
import numpy as np

def emotionImage(emotion):
    # Emojis
    if emotion == 'felicidad': image = cv2.imread('emojis/feliz.png')
    if emotion == 'enojo': image = cv2.imread('emojis/enojado.png')
    if emotion == 'sorpresa': image = cv2.imread('emojis/sorpresa.png')
    if emotion == 'tristeza': image = cv2.imread('emojis/triste.png')
    if emotion == 'neutral': image = cv2.imread('emojis/neutral.png')
    return image

def obtenerResultados(datos):
    sentimientos = [["enojo",0.0,0],["felicidad",0.0,0],["neutral",0.0,0],["sorpresa",0.0,0],["tristeza",0.0,0]]
    for dato in datos:
        for n in range(0, 4):
            if(dato[0] == n):
                sentimientos[n][2] += 1
                sentimientos[n][1] += dato[1]
    
    for n in range(0, 4):
        if(sentimientos[n][2] != 0):
            sentimientos[n][1] = sentimientos[n][1]/sentimientos[n][2]

    print(sentimientos)
        
# --------- Métodos usados para el entrenamiento y lectura del modelo ----------
#method = 'EigenFaces'
#method = 'FisherFaces'
method = 'LBPH'
if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()
emotion_recognizer.read('modelo'+method+'.xml')
# --------------------------------------------------------------------------------
dataPath = 'c:/users/casa/documents/python/reconocimiento_emociones/data' #Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)
cap = cv2.VideoCapture(2,cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

resultados = []

while True:
    ret,frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    # nFrame = cv2.hconcat([frame, np.zeros((480,300,3),dtype=np.uint8)])
    faces = faceClassif.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
        result = emotion_recognizer.predict(rostro)
        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
        # EigenFaces
        if method == 'EigenFaces':
            if result[1] < 5700:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                # image = emotionImage(imagePaths[result[0]])
                # nFrame = cv2.hconcat([frame,image])
            else:
                cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                # nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
        
        # FisherFace
        if method == 'FisherFaces':
            if result[1] < 500:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                # image = emotionImage(imagePaths[result[0]])
                # nFrame = cv2.hconcat([frame,image])
            else:
                cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                # nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
        
        # LBPHFace
        if method == 'LBPH':
            if result[1] < 60:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                resultados.append(result)
                
                # image = emotionImage(imagePaths[result[0]])
                # nFrame = cv2.hconcat([frame,image])
            else:
                cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                # nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
    cv2.imshow('nFrame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

obtenerResultados(resultados)
cap.release()
cv2.destroyAllWindows()
