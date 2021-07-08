# proyectoImagen - realreco

El sistema __realreco__ es una herramienta de reconocimiento facial con plataforma web que nos ofrece distintas ventajas.
- Localización y detección de personas a tiempo real
- Identificación de personas en la base de datos (Redis)
- Seguimiento de personas
- Visión web sencilla y accesible
- Ejecución a través de una cámara externa
## Developers
- Julio Robles
- Jorge de Santiago
- Luis Crespo
- Manuel Saravia

## Redis
Este código utiliza una base de datos Redis para el almacenamiento y recuperación de los encodings.
Por tanto, es necesario la instalación previa de un servidor redis. Se utiliza el puerto por defecto (port=6379) y la base de datos 1 (select 1).

## Instalación y puesta en marcha
Este programa está diseñado con librerías de instalación en Conda.
También recursos de CUDA.
Instalamos las librerias del enviroment.

```
conda install --file requirementsconda.txt
```
Migramos los cambios de Django del servidor
```
python manage.py makemigrations
python manage.py migrate
```
Iniciamos el programa
```
python manage.py runserver
```

## Funcionamiento
![alt text](https://github.com/julio-robles/proyectoImagen/blob/main/fotos/arch.png)
