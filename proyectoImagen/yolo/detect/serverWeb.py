import cv2
import socket
import base64
import numpy as np


HOST = '' 
PORT = 5555

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print("Listening on port " + str(PORT))
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            try: 
                frame = str(conn.recv(500000), 'utf-8')
                if not frame: 
                    break      
                if (len(frame) > 100000):
                    img = base64.b64decode(frame)
                    npimg = np.frombuffer(img, dtype=np.uint8)
                    source = cv2.imdecode(npimg, 1)
                    if (isinstance(source, type(np.array([1])))):
                        if (source.shape[0] > 1 and source.shape[1] > 1):
                            cv2.imshow('image', source)
                    cv2.waitKey(1)   
            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break