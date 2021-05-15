# Generated by Django 3.1.2 on 2021-05-15 17:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('yolo', '0004_auto_20210515_1920'),
    ]

    operations = [
        migrations.CreateModel(
            name='FaceDocument',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('faces', models.FileField(upload_to='faces')),
            ],
        ),
        migrations.AlterField(
            model_name='document',
            name='subida',
            field=models.FileField(upload_to='sin'),
        ),
    ]
