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
font = ''


def detect(opt, child_conn, lock, servSocket, save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
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

    last_ids = []
    actual_ids = []
    max_id = 0
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
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
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
                for *xyxy, conf, cls in reversed(det):    # Paralelizacion
                    # Face recognition
                    xywh = torch.tensor(xyxy).view(1, 4).tolist()
                    cropped_images.append([im0[int(xywh[0][1]):int(xywh[0][1]) + int(xywh[0][3]), int(xywh[0][0]):int(xywh[0][0]) + int(xywh[0][2])], int(xywh[0][0]), int(xywh[0][1])])
                    
                    p1 = (int(xywh[0][0]), int(xywh[0][1]))
                    p2 = (int(xywh[0][2]), int(xywh[0][3]))
                    color = (255, 0, 0)

                    person_id = -1
                    person_name = "???"

                    for person in range(len(last_box)):
                        l1 = (int(last_box[person][0][0]), int(last_box[person][0][1]))
                        l2 = (int(last_box[person][0][2]), int(last_box[person][0][3]))

                        dist1 = math.sqrt(((l1[0] - p1[0]) ** 2) + ((l1[1] - p1[1]) ** 2))
                        dist2 = math.sqrt(((l2[0] - p2[0]) ** 2) + ((l2[1] - p2[1]) ** 2))

                        if dist1+dist2 < 60:
                            person_id = last_ids[person]
                            person_name = last_names[person]
                            break
                        else:
                            # person_id = max(actual_ids, default=0)+1
                            max_id += 1
                            person_id = max_id
                            person_name = "???"

                            if max_id > 9999: 
                                max_id = 0

                    im0 = cv2.rectangle(im0, p1, p2, color, thickness=2)
                    cv2.putText(im0, person_name, (int(xywh[0][0]) + 2, int(xywh[0][1]) + 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, color,1)

                    #l_ids.append(person_id)
                    actual_box.append(xywh)
                    actual_ids.append(person_id)
                    actual_names.append(person_name)
            

                   

                # --------------------  AQUI USAREMOS UNA MP.PIPE PARA PASAR LA IMAGEN A FACE_RECOGNITION  --------------------
                
                
                last_box = []
                last_ids = []
                last_names = []

                for person in actual_box: last_box.append(person)
                for person in actual_ids: last_ids.append(person)
                for person in actual_names: last_names.append(person)


                lock.acquire()    
                try:
                    child_conn.send([cropped_images, actual_ids])
                    personas = child_conn.recv()    # siempre tiene que recivir
                    for i in range(len(personas[0])):                        
                        if (personas[0][i][0] != "???"):
                            indexPersons = last_ids.index(personas[1][i])
                            last_names[indexPersons] = personas[0][i][0]    
                finally:
                    actual_box = []
                    actual_ids = []
                    actual_names = []
                    lock.release()
                    


                
                

 
                


            # Print time (inference + NMS)
            #print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results into web
            if servSocket:
                servSocket
                servSocket.videoToServer(im0)

            
            
            if view_img:
                cv2.imshow(str(p), im0)
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
    PORT = 5555
    s = None
    def __init__(self):
        self.s = socket.socket()
        self.s.connect((self.HOST, self.PORT))

    def videoToServer(self, image):
        encoded, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer)
        self.s.sendall(jpg_as_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', default='0', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--createEncodings', type=str, default='0', help='create encodings 1, or not encodings 0')   
    parser.add_argument('--streamServer', type=str, default='0', help='stream to web server')   

    opt = parser.parse_args()
    print(opt)

    check_requirements(exclude=('pycocotools', 'thop'))
    
    if(opt.createEncodings == '1' or not os.path.isfile('dataset_faces.dat')):
        print('\nCreation of encodings\n')
        createEncondings()
    else:
        print('\nNo new creation of encodings\n')


    if(opt.streamServer == '1'):
        print('\nStream de video online\n')
        servSocket = Socket()
    else:
        servSocket = None
    
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
                
        else:
            parent_conn, child_conn = mp.Pipe()
            lock = mp.Lock()

            yoloProcess = mp.Process(target=detect, args=(opt, child_conn,  lock, servSocket,))
            yoloProcess.start()
            
            faceRecognitionProcess = mp.Process(target=faceRecognitionLoop, args=(parent_conn, lock,))
            faceRecognitionProcess.start()

            yoloProcess.join()
            faceRecognitionProcess.join()