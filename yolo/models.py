from django.db import models


class Document(models.Model):
    subida = models.FileField(upload_to='media/sin')

class FaceDocument(models.Model):
    faces = models.FileField(upload_to='faces')