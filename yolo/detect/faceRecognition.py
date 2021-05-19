import pickle
import cv2
import numpy as np
import glob
import face_recognition
from filters import *
from PIL import Image
import time

def createEncondings():  # Creamos los encodings si es necesario
    global nombres_conocidos
    nombres_conocidos = []
    global imagenes_deteccion
    imagenes_deteccion = []
    global encodings_conocidos
    encodings_conocidos = []

    # for filename in glob.glob('/samples/faces/*'):
    # Image.open(filename).convert('RGB').save('/samples/faces/' + filename.split('/')[-1].split('.')[0] + '.jpg')

    for filename in glob.glob('.\\samples\\faces\\*'):  # Carpeta donde se almacenan todas las imagenes
        # Lista con todos los nombres de las personas
        nombres_conocidos.append(filename.split('\\')[-1].split('.')[0])
        # Imagen de la persona
        image = face_recognition.load_image_file(filename)
        # Lista con todas las imagenes
        imagenes_deteccion.append(image)
        # Lista con los encodings de las imagenes                                                     
        encodings_conocidos.append(face_recognition.face_encodings(image)[0])

        # Diccionario de los encodings con el nombre de la persona a almacenar
    all_face_encodings = {}
    count = 0
    for name in nombres_conocidos:
        all_face_encodings[name] = encodings_conocidos[count]
        count += 1
    # Almacenamos en el fichero los encodings de las caras de las personas
    with open('./dataset_faces.dat', 'wb') as f:
        pickle.dump(all_face_encodings, f)


def loadEncondings():
    with open('./dataset_faces.dat', 'rb') as f:
        all_face_encodings = pickle.load(f)

    # Recogemos los nombres y los encodings
    global nombres_conocidos
    nombres_conocidos = list(all_face_encodings.keys())
    global encodings_conocidos
    encodings_conocidos = np.array(list(all_face_encodings.values()))
    print('Users in encodigns:\n' + str(nombres_conocidos) + '\n')


def faceRecognitionLoop(queueFace, queueYolo):
    loadEncondings()
    newConn = 1
    while True:
        if not queueFace.empty():
            # print("face_recognition")
            newConn = queueFace.get()
            persons = []
            for person in newConn[0]:
                nombre = faceRecognition(person[0])
                if nombre:
                    persons.append(nombre)
                else:
                    persons.append(['???'])
            persons2 = []
            persons2.append(persons)
            persons2.append(newConn[1])
            queueYolo.put(persons2)
            




    



def faceRecognition(face):
    # img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    img = face
    # Definir tres arrays, que servirán para guardar los parámetros de los rostros que se encuentren en la imagen:
    loc_rostros = []  # Localizacion de los rostros en la imagen (contendrá las coordenadas de los recuadros que las contienen)
    encodings_rostros = []  # Encodings de los rostros
    nombres_rostros = []  # Nombre de la persona de cada rostro

    #img_blurred = blur(img)

    # Localizamos cada rostro de la imagen y extraemos sus encodings:
    loc_rostros = face_recognition.face_locations(img, model='hog')


    if (len(loc_rostros) > 0):
        crop_face = face[loc_rostros[0][0]:loc_rostros[0][2], loc_rostros[0][3]:loc_rostros[0][1]]


    #img_filter = contrast(img)
    #img_filter = brightness(img_filter)
    #img_filter = sharp(img_filter)

    encodings_rostros = face_recognition.face_encodings(img, loc_rostros)

    # Recorremos el array de encodings que hemos encontrado:
    for encoding in encodings_rostros:

        # Buscamos si hay alguna coincidencia con algún encoding conocido:
        coincidencias = face_recognition.compare_faces(encodings_conocidos, encoding, tolerance=0.5)  # Paralelizar

        # El array 'coincidencias' es ahora un array de booleanos.
        # Si contiene algun 'True', es que ha habido alguna coincidencia:
        if True in coincidencias:
            # Buscamos el nombre correspondiente en el array de nombres conocidos:
            nombre = nombres_conocidos[coincidencias.index(True)]

        # Si no hay ningún 'True' en el array 'coincidencias', no se ha podido identificar el rostro:
        else:
            nombre = "???"

        # Añadimos el nombre de la persona identificada en el array de nombres:
        nombres_rostros.append(nombre)

    return nombres_rostros
