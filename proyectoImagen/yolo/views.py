from yolo import detect
from django.http.response import StreamingHttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import socket
import base64
import numpy as np
import cv2
import subprocess



# Create your views here.
def webcam():
    detect.bridge('0', 'runs/webcam','exp')

def home(request):
    # webcam()
    # if request.method == 'POST':
    #     archivo = request.FILES['file']
    #     fs = FileSystemStorage()
    #     print(archivo.name)
    #     fs.save(archivo.name, archivo)
    #     file = 'data/images/' + archivo.name
    #     detect.bridge(file, 'runs/video', archivo.name)
    
    return render(request, 'yolo/home.html', {'title': 'Home'})


def ip(request):
    # detect.bridge()
    return render(request, 'yolo/ip.html', {'title': 'Ip'})

def videoToSend():
	HOST = '127.0.0.1' 
	PORT = 8080
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind((HOST, PORT))
		s.listen(10)
		print("Listening on port " + str(PORT))
		subprocess.Popen("python yolo/detect/detect.py", shell=True)
		conn, addr = s.accept()
		with conn:
			print('Connected by', addr)
			while True:
				try: 
					frame = str(conn.recv(300000), 'utf-8')
					if not frame: 
						break      
					#if (len(frame) > 100000):
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

def video(request):
	return StreamingHttpResponse(videoToSend(), content_type='multipart/x-mixed-replace; boundary=frame')
	

"""
class IPWebCam(object):
	def __init__(self):
		self.url = "http://192.168.0.100:8080/shot.jpg"

	def __del__(self):
		cv2.destroyAllWindows()

	def get_frame(self):
		imgResp = urllib.request.urlopen(self.url)
		imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
		img= cv2.imdecode(imgNp,-1)
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces_detected = face_detection_webcam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
		for (x, y, w, h) in faces_detected:
			cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
		resize = cv2.resize(img, (640, 480), interpolation = cv2.INTER_LINEAR) 
		frame_flip = cv2.flip(resize,1)
		ret, jpeg = cv2.imencode('.jpg', frame_flip)
		return jpeg.tobytes()
"""