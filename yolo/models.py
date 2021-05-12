from django.db import models


class Document(models.Model):
    subida = models.FileField(upload_to='sin')
    ruta = models.TextField