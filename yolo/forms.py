from django import forms


class DocumentForm(forms.Form):
    subida = forms.FileField(label='Seleccione un archivo para detectar')
class IpForm(forms.Form):
    ip = forms.CharField(label="",required=True)

class IdForm(forms.Form):
    img = forms.FileField(label='Seleccione una imagen de una persona', required=True)