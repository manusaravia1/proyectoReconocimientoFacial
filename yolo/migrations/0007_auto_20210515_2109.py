# Generated by Django 3.1.2 on 2021-05-15 19:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('yolo', '0006_auto_20210515_1942'),
    ]

    operations = [
        migrations.AlterField(
            model_name='facedocument',
            name='faces',
            field=models.FileField(upload_to='samples/faces'),
        ),
    ]
