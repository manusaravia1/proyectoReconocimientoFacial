from django import forms


class DocumentForm(forms.Form):
    subida = forms.FileField(label='Seleccione un archivo')
