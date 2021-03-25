import cv2
import face_recognition

"""
def getfiles():
    for root, dirs, files in os.walk("."):
        for name in files:
            yield os.path.join(root, name)
files = filter(lambda image: (image[-3:] == '.jpg' or image[-3:] == '.png'), getfiles())
# now you can loop your image detection api call.
for file in files
    image = face_recognition.load_image_file(file)
    face_locations = face_recognition.face_locations(image)
    print(face_locations)
"""


def createEncondings():
    # Cargamos las imagenes con los rostros que queremos identificar:

    imagen_hulio = face_recognition.load_image_file('samples/faces/hulio.jpeg')
    # imagen_julioMask = face_recognition.load_image_file('samples/faces/julioMask.jpeg')
    imagen_horhe = face_recognition.load_image_file('samples/faces/horhe.jpeg')
    imagen_manu = face_recognition.load_image_file('samples/faces/manu.jpeg')
    imagen_rafa = face_recognition.load_image_file('samples/faces/rafa.jpeg')

    # El siguiente paso es extraer los 'encodings' de cada imagen.
    # Los encodings son las características únicas de cada rostro que permiten diferenciarlo de otros.
    hulio_encodings = face_recognition.face_encodings(imagen_hulio)[0]
    # julioMask_encodings = face_recognition.face_encodings(imagen_julioMask)[0]
    horhe_encodings = face_recognition.face_encodings(imagen_horhe)[0]
    manu_encodings = face_recognition.face_encodings(imagen_manu)[0]
    rafa_encodings = face_recognition.face_encodings(imagen_rafa)[0]

    # Creamos un array con los encodings y otro con sus respectivos nombres:
    encodings_conocidos = [
        hulio_encodings,
        horhe_encodings,
        manu_encodings,
        rafa_encodings
    ]
    nombres_conocidos = [
        "Julio",
        "Jorhe",
        "Manu",
        "Rafa"
    ]

    font = cv2.FONT_HERSHEY_COMPLEX

    return [encodings_conocidos, nombres_conocidos, font]


def faceRecognition(face, encodings_conocidos, nombres_conocidos, font):
    img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # Definir tres arrays, que servirán para guardar los parámetros de los rostros que se encuentren en la imagen:
    loc_rostros = []  # Localizacion de los rostros en la imagen (contendrá las coordenadas de los recuadros que las contienen)
    encodings_rostros = []  # Encodings de los rostros
    nombres_rostros = []  # Nombre de la persona de cada rostro

    # Localizamos cada rostro de la imagen y extraemos sus encodings:
    loc_rostros = face_recognition.face_locations(img)
    encodings_rostros = face_recognition.face_encodings(img, loc_rostros)

    # Recorremos el array de encodings que hemos encontrado:
    for encoding in encodings_rostros:

        # Buscamos si hay alguna coincidencia con algún encoding conocido:
        coincidencias = face_recognition.compare_faces(encodings_conocidos, encoding)

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