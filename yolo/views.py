from yolo import detect
from django.http.response import StreamingHttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import redirect, render
from django.core.files.storage import FileSystemStorage
from django.views.generic import CreateView
import socket
import base64
import numpy as np
import cv2
import subprocess
import signal
import os
import shutil
import time
import platform

from .models import Document, FaceDocument
from .forms import DocumentForm, IpForm, IdForm
from .detect.redisDbase import rdb



class IPWebCam():
	def __init__(self):	
		self.HOST = '127.0.0.1' 
		self.PORT = 8080

	def __del__(self):
		print("Terminamos la detección e identificación...")
		print(platform.system())
		if (platform.system() == 'Windows'):
			os.kill(self.proceso.pid, signal.CTRL_BREAK_EVENT)
		else:
			self.proceso.terminate()

	def get_frame(self, ipwebcam):
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			s.bind((self.HOST,self.PORT))
			s.listen(10)
			print("Listening on port " + str(self.PORT))
			llamada = "python yolo/detect/detect.py --source " + ipwebcam + " --streamServer 1"
			self.proceso = subprocess.Popen(llamada.split(), shell=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
			conn, addr = s.accept()
			with conn:
				print('Connected by', addr)
				while True:
					try: 
						frame = str(conn.recv(300000), 'utf-8')
						if not frame: 
							break
						img = base64.b64decode(frame)
						npimg = np.frombuffer(img, dtype=np.uint8)
						source = cv2.imdecode(npimg, 1)
						if (isinstance(source, type(np.array([1])))):
							if (source.shape[0] > 1 and source.shape[1] > 1):
								yield (b'--frame\r\n'
										b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n')
							cv2.waitKey(1)   
					except KeyboardInterrupt:
						cv2.destroyAllWindows()
						break

def home(request):
	return render(request, 'home.html', {'title': 'Home'})

def ip(request):
	if request.method=='POST':
		form = IpForm(request.POST)
		if form.is_valid():
			return render(request, 'ip.html', {'title': 'Home','ip': form.cleaned_data['ip'], 'form':IpForm()})
	else:
		return render(request, 'ip.html', {'title': 'Ip', 'form':IpForm()})

def upload(request):
	message = 'Bien'
	if request.method == 'POST':
		if os.path.exists("media/exp/"):
			shutil.rmtree("media/exp/")
		form = DocumentForm(request.POST, request.FILES)
		if form.is_valid():
			newfile = Document(subida=request.FILES['subida'])
			nombre = request.FILES['subida'].name
			newfile.save()
			llamada = "python yolo/detect/detect.py --source media/sin/" + nombre
			subprocess.run(llamada, shell=True)
			if os.path.exists("media/sin/"):
				shutil.rmtree("media/sin/")
			detectado_url = "media/exp/" + nombre
			context={'detectado_nombre':nombre,'detectado_url':detectado_url, 'form':form,'message':message}
			return render(request, 'list.html', context)
		else:
			message='Error'
	else:
		form = DocumentForm()

	context={'form':form,'message':message}
	return render(request, 'list.html', context)


def id(request):
	if request.method == 'POST':
		# Vaciamos la db redis
		rdb.empty()

		form = IdForm(request.POST, request.FILES)
		if form.is_valid():
			FaceDocument(faces=request.FILES['img']).save()
			return redirect ('home')
	
	return render(request, 'id.html',{'form':IdForm()})



def video(request, ip):
	if ip:
		d = IPWebCam()
		return StreamingHttpResponse(d.get_frame(ip), content_type='multipart/x-mixed-replace; boundary=frame')
