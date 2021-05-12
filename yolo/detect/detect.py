import argparse
import time
from pathlib import Path
import multiprocessing as mp
import os
import socket
import base64

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import math

from faceRecognition import *

imagenes_deteccion = []
encodings_conocidos = []
nombres_conocidos = []
font = cv2.FONT_HERSHEY_COMPLEX


def detect(obj_source, child_conn, lock, servSocket, save_img=False):
    source, weights, view_img, save_txt, imgsz = obj_source, 'yolov5s.pt', False, False, 640
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path("media\exp")  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    last_box = []
    actual_box = []
    actual_distances = []
    actual_coordinates = []

    last_ids = []
    actual_ids = []
    max_id = 0

    last_dir = []
    actual_dir = []

    last_names = []
    actual_names = []

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.35, 0.45, classes=0, agnostic=False)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                cropped_images = []
                for *xyxy, conf, cls in reversed(det):  # Paralelizacion
                    # Face recognition
                    xywh = torch.tensor(xyxy).view(1, 4).tolist()
                    cropped_images.append([im0[int(xywh[0][1]):int(xywh[0][1]) + int(xywh[0][3]),
                                           int(xywh[0][0]):int(xywh[0][0]) + int(xywh[0][2])], int(xywh[0][0]),
                                           int(xywh[0][1])])

                    p1 = (int(xywh[0][0]), int(xywh[0][1]))
                    p2 = (int(xywh[0][2]), int(xywh[0][3]))

                    person_id = -1
                    person_name = "???"
                    direction = (0, 0)
                    distance_to_box = -1

                    for person in range(len(last_box)):
                        l1 = (int(last_box[person][0][0]), int(last_box[person][0][1]))
                        l2 = (int(last_box[person][0][2]), int(last_box[person][0][3]))

                        center1 = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
                        center2 = (int((l1[0] + l2[0]) / 2), int((l1[1] + l2[1]) / 2))
                        center3 = (center2[0] + last_dir[person][0], center2[1] + last_dir[person][1])

                        dist = math.sqrt(((center1[0] - center3[0]) ** 2) + ((center1[1] - center3[1]) ** 2))

                        if dist < im0.shape[1]/5:

                            if last_names[person] not in actual_names:

                                person_id = last_ids[person]
                                person_name = last_names[person]
                                distance_to_box = dist

                                direction = (int(center1[0] - center2[0]), int(center1[1] - center2[1]))
                                break

                            else:

                                other_index = actual_names.index(last_names[person])

                                if dist < actual_distances[other_index]:

                                    person_id = last_ids[person]
                                    person_name = last_names[person]
                                    distance_to_box = dist

                                    direction = (int(center1[0] - center2[0]), int(center1[1] - center2[1]))

                                    actual_box.pop(other_index)
                                    actual_ids.pop(other_index)
                                    actual_names.pop(other_index)
                                    actual_dir.pop(other_index)
                                    actual_distances.pop(other_index)

                                else:
                                    max_id += 1
                                    person_id = max_id
                                    person_name = "???"

                                    if max_id > 9999: max_id = 0

                                break

                        else:
                            # person_id = max(actual_ids, default=0)+1
                            max_id += 1
                            person_id = max_id
                            person_name = "???"

                            if max_id > 9999: max_id = 0


                    actual_box.append(xywh)
                    actual_ids.append(person_id)
                    actual_names.append(person_name)
                    actual_dir.append(direction)
                    actual_distances.append(distance_to_box)
                    actual_coordinates.append(xywh[0])

                for person in range(len(actual_ids)):
                    if (actual_names[person] != "???"):
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)

                    p1 = (int(actual_coordinates[person][0]), int(actual_coordinates[person][1]))
                    p2 = (int(actual_coordinates[person][2]), int(actual_coordinates[person][3]))

                    im0 = cv2.rectangle(im0, p1, p2, color, thickness=2)
                    cv2.putText(im0, actual_names[person], (int(actual_coordinates[person][0]) + 2, int(actual_coordinates[person][1]) + 20), font, 0.5, color, 1)
                    cv2.putText(im0, str(actual_dir[person]), (int(actual_coordinates[person][2]) - 80, int(actual_coordinates[person][1]) + 20), font, 0.5, color, 1)

                # --------------------  AQUI USAREMOS UNA MP.PIPE PARA PASAR LA IMAGEN A FACE_RECOGNITION  --------------------

                last_box = []
                last_ids = []
                last_names = []
                last_dir = []
                actual_coordinates = []

                for person in actual_box: last_box.append(person)
                for person in actual_ids: last_ids.append(person)
                for person in actual_names: last_names.append(person)
                for person in actual_dir: last_dir.append(person)

                lock.acquire()
                try:
                    child_conn.send([cropped_images, actual_ids])
                    personas = child_conn.recv()  # siempre tiene que recibir
                    for i in range(len(personas[0])):
                        if personas[0][i][0] != "???":
                            indexPersons = last_ids.index(personas[1][i])
                            last_names[indexPersons] = personas[0][i][0]
                finally:
                    actual_box = []
                    actual_ids = []
                    actual_names = []
                    actual_dir = []
                    lock.release()

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results into web
            if servSocket:
                # servSocket
                servSocket.videoToServer(im0)
                
            if view_img:
                # cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
                

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            cv2.destroyAllWindows()
            break

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

class Socket:
    HOST = '127.0.0.1'
    PORT = 8080
    s = None

    def __init__(self):
        self.s = socket.socket()
        self.s.connect((self.HOST, self.PORT))

    def videoToServer(self, image):
        try:    
            encoded, buffer = cv2.imencode('.jpg', image)
            jpg_as_text = base64.b64encode(buffer)
            self.s.sendall(jpg_as_text)
        except:
            pass
            #print("Fallo en la conexion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--streamServer', type=str, default='0', help='source')
    parser.add_argument('--createEncodings', type=str, default='0', help='source')
    opt = parser.parse_args()

    check_requirements(exclude=('pycocotools', 'thop'))

    if opt.createEncodings == '1' or not os.path.isfile('dataset_faces.dat'):
        print('\nCreation of encodings\n')
        createEncondings()
    else:
        print('\nNo new creation of encodings\n')

    if opt.streamServer == '1':
        print('\nStream de video online\n')
        servSocket = Socket()
    else:
        servSocket = None
    
    with torch.no_grad():    
        parent_conn, child_conn = mp.Pipe()
        lock = mp.Lock()

        yoloProcess = mp.Process(target=detect, args=('0', child_conn, lock, servSocket,))
        yoloProcess.start()

        faceRecognitionProcess = mp.Process(target=faceRecognitionLoop, args=(parent_conn, lock,))
        faceRecognitionProcess.start()
        yoloProcess.join()
        print("-------------------------yolo")
        faceRecognitionProcess.terminate()
        faceRecognitionProcess.join()
        print("-------------------------face")

    # if opt.source == '0':
    #     opt.source = 'http://192.168.246.170:8080/video'


# def bridge(obj_source = '0', createEncodings = '0', streamServer = '0'):
#     check_requirements(exclude=('pycocotools', 'thop'))

#     if createEncodings == '1' or not os.path.isfile('dataset_faces.dat'):
#         print('\nCreation of encodings\n')
#         createEncondings()
#     else:
#         print('\nNo new creation of encodings\n')

#     if streamServer == '1':
#         print('\nStream de video online\n')
#         servSocket = Socket()
#     else:
#         servSocket = None
    
#     with torch.no_grad():    
#         parent_conn, child_conn = mp.Pipe()
#         lock = mp.Lock()

#         yoloProcess = mp.Process(target=detect, args=(obj_source, child_conn, lock, servSocket,))
#         yoloProcess.start()

#         faceRecognitionProcess = mp.Process(target=faceRecognitionLoop, args=(parent_conn, lock,))
#         faceRecognitionProcess.start()

#         yoloProcess.join()
#         faceRecognitionProcess.join()
    
#     return