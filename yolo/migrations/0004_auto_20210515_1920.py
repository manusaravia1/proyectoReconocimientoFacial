# Generated by Django 3.1.2 on 2021-05-15 17:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('yolo', '0003_auto_20210511_1813'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='subida',
            field=models.FileField(upload_to='pan'),
        ),
    ]
