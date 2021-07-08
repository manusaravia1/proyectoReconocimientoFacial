import glob
import face_recognition
import time
import numpy as np
from redisDbase import rdb


def createEncondings():  # Creamos los encodings si es necesario
    inicio = time.time()
    global nombres_conocidos
    nombres_conocidos = []
    global encodings_conocidos
    encodings_conocidos = []

    for filename in glob.glob('.\\samples\\faces\\*'):  # Carpeta donde se almacenan todas las imagenes
        # De la carpeta tomo el nombre, foto y calculo encodings
        nombre = filename.split('\\')[-1].split('.')[0]
        image = face_recognition.load_image_file(filename)
        encodings = face_recognition.face_encodings(image)[0]
        # To REDIS: key = nombre, value = encodings
        rdb.addEncoding(nombre, encodings)

    print(">>> createEncodings({}) time: {}".format(rdb.howMany(), time.time() - inicio))


def loadEncondings():
    inicio = time.time()
    #print(">>> loadEncondings...({})".format(rdb.howMany()))
    global nombres_conocidos
    nombres_conocidos = rdb.getAllKeys() # en modo lista de strings
    global encodings_conocidos
    encodings_conocidos  = np.array(rdb.getAllEncodings())  # a formato numpy array
    print(">>> loadEncodings({}) time: {}".format(rdb.howMany(), time.time() - inicio))


def faceRecognitionLoop(queueFace, queueYolo):
    loadEncondings()
    newConn = 1
    while True:
        if not queueFace.empty():
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
    inicio = time.time()
    match = False
    img = face
    # Definimos tres arrays: Localizacion, Encodings y Nombres
    loc_rostros = []  # Localizacion de los rostros en la imagen
    encodings_rostros = []  # Encodings de los rostros
    nombres_rostros = []  # Nombre de la persona de cada rostro

    # Localizamos cada rostro de la imagen y extraemos sus encodings:
    loc_rostros = face_recognition.face_locations(img, model='hog')

    if (len(loc_rostros) > 0):
        crop_face = face[loc_rostros[0][0]:loc_rostros[0][2], loc_rostros[0][3]:loc_rostros[0][1]]

    encodings_rostros = face_recognition.face_encodings(img, loc_rostros)

    # Recorremos el array de encodings que hemos encontrado:
    for encoding in encodings_rostros:
        # Buscamos si hay alguna coincidencia con algún encoding conocido. Paralelizar
        coincidencias = face_recognition.compare_faces(encodings_conocidos, encoding, tolerance=0.5)
        # El array 'coincidencias' es ahora un array de booleanos.
        # Si contiene algun 'True', es que ha habido alguna coincidencia:
        if True in coincidencias:
            # Nos quedamos con el primer nombre coincidente:
            nombre = nombres_conocidos[coincidencias.index(True)]
            match = True
        # Si no hay ningún 'True' en el array 'coincidencias', no se ha podido identificar el rostro:
        else:
            nombre = "???"

        # Añadimos el nombre de la persona identificada en el array de nombres:
        nombres_rostros.append(nombre)

    #print("\n>>> faceRecognition() match = {}, time: {}".format(match, time.time() - inicio))
    return nombres_rostros
