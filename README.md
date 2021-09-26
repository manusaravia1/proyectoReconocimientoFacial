# proyectoImagen - realreco

El sistema __realreco__ es una herramienta de reconocimiento facial con plataforma web que nos ofrece distintas ventajas.
- Localización y detección de personas a tiempo real
- Identificación de personas en la base de datos
- Seguimiento de personas
- Visión web sencilla y accesible
- Ejecución a través de una cámara externa

- Branch "main_RT_redis": mejora del rendimiento del proyecto existente incorporando una base de datos Redis para almacenar los encodings y acceder a ellos directamente desde Redis durante el proceso de identificación o “matching” entre una nueva imagen (cara) y el conjunto de encodings almacenados.
 
## Developers
- Julio Robles
- Manuel Saravia
- Jorge de Santiago
- Luis Crespo

## Instalación y puesta en marcha
Este programa está diseñado con librerías de instalación en Conda.
También recursos de CUDA.
Instalamos las librerias del enviroment.

```
conda install --file requirementsconda.txt
```
Migranos los cambios de Django del servidor
```
python manage.py makemigrations
python manage.py migrate
```
Iniciamos el programa
```
python manage.py runserver
```
Para el Branch "main_RT_redis" es necesario instalar Redis así como la librería redis de Python.
