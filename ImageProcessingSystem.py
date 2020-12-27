import cv2
import dlib
import time
import threading
import math
import os
import tensorflow as tf
import numpy as np
import label_map_util
import rticonnextdds_connector as rti
import importlib.util
import Plate_recog.ssd_plate_detection.tf_detect as lp_recog
import pytesseract

from sys import path as sys_path
from os import path as os_path
file_path = os_path.dirname(os_path.realpath(__file__))
sys_path.append(file_path + "/../../../")
import base64


WIDTH = 1280
HEIGHT = 720
carWidth = 1.85

# def carNumber(carNum, cID):
    # time.sleep(2)
    # carNum[cID] = 'Car ' + str(cID)
# ALPR-hez használt függvény a recognize_ndarray() és egy json objektumot ad vissza.
ppmList= []
ppmLicensePlate = []

def estimateSpeed(location1, location2, videoFps,fps):
    # Divide the original fps of the video with the current fps to get a better
    # prediction of the speed
    duration = abs(location2[4] - location1[4]) / fps
    adjustment = fps / videoFps
    # Euclidean distance
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = location2[2] / carWidth
    ppmList.append(ppm)
   # if(len(ppmLicensePlate) > 1):
      #  d_meters = d_pixels / np.median(ppmLicensePlate)
    #else:
    d_meters = d_pixels / np.median(ppmList)
    speed = d_meters / duration * 3.6
    return speed

# Modellek: 
#   - ssd_mobilenet_v1_coco_.... -> Kb legjobb teljesítményt adja, elég gyors és a predikciók sem rosszak
#   - ssd_mobilenet_v1_fpn_shared_box_predictor_640x640 -> Lassú nagyon, viszont elég pontos néha az alacsony
#     fps miatt túl sokszor felismeri ugyanazt az objektumot, de ez a lassú fps miatt van.
#   - ssd_mobilenet_v2_oid_v4 -> Jó volt, talán lassabb mint a coco-s változata és nyögvenyelősen lehet csak
#     megoldani az osztályozást
#   - ssd_resnet... -> szuper lassú
#   - ssdlite_modiledet_cpu -> valszeg ez telefonra van, mindenesetre itt nem sikerült működésre bírni
#   - ssdlite_mobilenet_v2_coco -> Jól működik, elég fine tuningal elég jó is lehet, bár nincs sok különbség a
#     mobilenet_v1_coco-val összehasonlítva

CWD_PATH = os.getcwd()

def trackMultipleObjects():
    MODEL_NAME = './models/ssdlite_mobilenet_v2_coco_2018_05_09'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = \
    'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90

    lp_detection_num = None
    lp_detection_scores = None
    lp_detection_boxes = None
    lp_detection_classes = None

    detection_graph = tf.Graph()
    LP_detection_graph = tf.Graph()
    detection_sess = tf.Session(graph=detection_graph)
    LP_detection_sess = tf.Session(graph=LP_detection_graph)

    LP_detection_graph_def = tf.GraphDef()
    detection_graph_def = tf.GraphDef()
    
    # Import the TF graph : first (Vehicle Detection)
    with detection_sess.as_default():
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    # Import the TF graph : second (Plate recognition)
    with LP_detection_sess.as_default():
        with LP_detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile('./Plate_recog/ssd_plate_detection/model/frozen_inference_graph.pb', 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            lp_detection_num = LP_detection_sess.graph.get_tensor_by_name('num_detections:0')
            lp_detection_boxes = LP_detection_sess.graph.get_tensor_by_name('detection_boxes:0')
            lp_detection_scores = LP_detection_sess.graph.get_tensor_by_name('detection_scores:0')
            lp_detection_classes = LP_detection_sess.graph.get_tensor_by_name('detection_classes:0')
            

    rectangleColor = (0, 0, 255)
    frameCounter = 0
    currentCarID = 0
    fps = 0

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    # Dictionaries for keeping up with the state of the tracked vehicles
    realVehicleClasses = {}
    carTracker = {}
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    vehicleSpeeds = {}
    vehicleLicenseplates = {}
    
    # RTI Connector for VehicleData
    vehicleDataConnector = rti.Connector("MyParticipantLibrary::MyParticipant","VehicleData.xml")
    output = vehicleDataConnector.get_output("MyPublisher::MyVehicleDataWriter")

    
     # Definite input and output Tensors for detection_graph
    with detection_sess.as_default():
        with detection_graph.as_default():
            image_tensor = detection_sess.graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_sess.graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_sess.graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_sess.graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_sess.graph.get_tensor_by_name('num_detections:0')
            
            
            with rti.open_connector(
                config_name="MyParticipantLibrary::MySubParticipant",
                url=file_path + "/ImageType.xml") as connector:
                
                input = connector.get_input("MySubscriber::MyImageReader")
                input.wait_for_publications()
                
                while True:
                    input.wait()
                    input.take()
                    for sample in input.samples.valid_data_iter:
                        start_time = time.time()
                        frame = sample.get_string("data")
                        videoFps = sample.get_number("videoFps")
                        frameToByte = frame.encode("utf-8")
                        decoded = base64.b64decode(frameToByte)
                        np_data = np.frombuffer(decoded,dtype=np.uint8)
                        image = cv2.imdecode(np_data,flags=1)
    
                        image_np_expanded = np.expand_dims(image, axis=0)
                        resultImage = image.copy()
                        
                        frameCounter = frameCounter + 1
                        
                        carIDtoDelete = []
                        for carID in carTracker.keys():
                            trackingQuality = carTracker[carID].update(image)
                            
                            if trackingQuality < 6:
                                carIDtoDelete.append(carID)
                                
                        for carID in carIDtoDelete:
                            print('Removing carID ' + str(carID) + ' from list of trackers.')
                            print ('Removing carID ' + str(carID) + ' previous location.')
                            print ('Removing carID ' + str(carID) + ' current location.')
                            carTracker.pop(carID, None)
                            carLocation1.pop(carID, None)
                            carLocation2.pop(carID, None)
                        
                        if not (frameCounter % 7):
                            (boxes, scores, classes, num) = \
                            detection_sess.run([detection_boxes, detection_scores,
                            detection_classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})
                            boxes = np.squeeze(boxes) # Location sorrend: ymin,xmin,ymax,xmax
                            scores = np.squeeze(scores)
                            classes = np.squeeze(classes)
                            cars = []
                            for i in range(len(scores)):
                                if(scores[i] > 0.5):
                                    if classes[i] in category_index.keys():
                                        class_name = category_index[classes[i]]['name']
                                    if(class_name == "train"):
                                        class_name = "car"
                                    if(class_name == "cell phone"):
                                        class_name = "car"
                                    if(class_name == "boat"):
                                        class_name = "truck"
                                    if class_name == "truck" or class_name == "car" or class_name == "bus" or class_name == "motorcycle":
                                        car = tuple(boxes[i].tolist())
                                        cars.append([car,class_name])
                            for (coords,class_name) in cars:
                                (_y, _x, _h, _w) = coords
                                x = int(_x * WIDTH) #xmin
                                y = int(_y * HEIGHT) #ymin
                                w = int(_w * WIDTH) - x #xmax
                                h = int(_h * HEIGHT) - y #ymax
                                x_bar = x + 0.5 * w
                                y_bar = y + 0.5 * h
                                
                                matchCarID = None
                            
                                for carID in carTracker.keys():
                                    trackedPosition = carTracker[carID].get_position()
                                    
                                    t_x = int(trackedPosition.left())
                                    t_y = int(trackedPosition.top())
                                    t_w = int(trackedPosition.width())
                                    t_h = int(trackedPosition.height())
                                    
                                    t_x_bar = t_x + 0.5 * t_w
                                    t_y_bar = t_y + 0.5 * t_h
                                    
                                    
                                    # Checking if we already have this vehicle tracked
                                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                                        matchCarID = carID
                                        realVehicleClasses[matchCarID] = class_name
                                # Creating a new tracker for a newly detected vehicle.
                                if matchCarID is None:
                                    realVehicleClasses[currentCarID] = class_name
                                    vehicleSpeeds[currentCarID] = "N/A"
                                    vehicleLicenseplates[currentCarID] = "N/A"
                                    print ('Creating new tracker ' + str(currentCarID))
                                    print('Vehicle type: ' + realVehicleClasses[currentCarID])
                                    cropped = image[y:y+h,x:x+w]
                                    lp_text = "N/A"
                                    with LP_detection_sess.as_default():
                                        with LP_detection_graph.as_default():
                                            lp_x,lp_y,lp_w,lp_h,score = lp_recog.detect(cropped,LP_detection_sess,lp_detection_num,lp_detection_scores,lp_detection_boxes,lp_detection_classes)
                                            if(lp_x != -1 ):
                                                licensePlate = cropped[lp_y:lp_h,lp_x:lp_w]
                                                # American Licenseplate is 305 mm wide, but European is 520 mm
                                                ppmLicensePlate.append(lp_w / 0.305 )
                                                cv2.imshow('License Plate',licensePlate)
                                                config = ('--oem 1 --psm 13')
                                                lp_text = pytesseract.image_to_string(licensePlate,config=config)
                                                vehicleLicenseplates[currentCarID] = lp_text
                                                cv2.putText(cropped, lp_text ,(10,35),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 0xFF),2,cv2.FONT_HERSHEY_SIMPLEX)
                                            if cropped.shape[0] != 0 or cropped.shape[1] != 0:
                                                cv2.imshow('Detected Cars',cropped)
                                    output.instance.set_string("type",class_name)
                                    output.instance.set_string("licensePlate",lp_text)
                                    output.instance.set_number("speed",-1)
                                    output.instance.set_number("vehicleID",currentCarID)
                                    output.write()
                                    
                                    #print(json.dumps(results,indent=4))
                                    #print('License Plate for ' + str(currentCarID) + ' is: ')
                                    tracker = dlib.correlation_tracker()
                                    #dlib.rectangle(left,top,right,bottom)
                                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                                    
                                    carTracker[currentCarID] = tracker
                                    carLocation1[currentCarID] = [x, y, w, h,frameCounter]
                                    # t = threading.Thread(target = carNum, args = (carNumbers, currentCarID))
                                    # t.start()
                                    currentCarID = currentCarID + 1                                   
                                    
                        for carID in carTracker.keys():
                            trackedPosition = carTracker[carID].get_position()
                                    
                            t_x = int(trackedPosition.left())
                            t_y = int(trackedPosition.top())
                            t_w = int(trackedPosition.width())
                            t_h = int(trackedPosition.height())
                            
                            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 2)
                            
                            # speed estimation
                            carLocation2[carID] = [t_x, t_y, t_w, t_h,frameCounter]
                            
                            cv2.putText(resultImage, realVehicleClasses[carID] + " " + str(vehicleSpeeds[carID]), (int(t_x + t_w/2), int(t_y)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                            
                            if(frameCounter % 15 == 0):
                                output.instance.set_number("vehicleID",carID)
                                if(vehicleSpeeds[carID] == "N/A"):
                                    output.instance.set_number("speed",-1)
                                else:
                                    if(math.isnan(vehicleSpeeds[carID])):
                                        vehicleSpeeds[carID] = -1
                                    output.instance.set_number("speed",int(vehicleSpeeds[carID]))
                                output.instance.set_string("type",realVehicleClasses[carID])
                                output.instance.set_string("licensePlate",vehicleLicenseplates[carID])
                                output.write()
                        
                        end_time = time.time()
                        if not (end_time == start_time):
                            fps = 1.0/(end_time - start_time)
                        cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                        for i in carLocation1.keys():
                            if frameCounter % 10 == 0:
                                [x1, y1, w1, h1,frameNumber1] = carLocation1[i]
                                [x2, y2, w2, h2,frameNumber2] = carLocation2[i]
                                # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
                                carLocation1[i] = [x2, y2, w2, h2,frameNumber2]
                                # print 'new previous location: ' + str(carLocation1[i])
                                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                                    speed = estimateSpeed([x1, y1, w1, h1,frameNumber1], [x2, y2, w2, h2,frameNumber2], videoFps, fps)
                                    vehicleSpeeds[i] = round(speed,2)
                                    print ('CarID ' + str(i) + ' speed is ' + str("%.2f" % round(speed, 2)) + ' km/h.\n')
                        cv2.imshow('result', resultImage)
                                
                    if cv2.waitKey(33) == 27:
                        vehicleDataConnector.close()
                        break
    
    vehicleDataConnector.close()                
    cv2.destroyAllWindows()


if __name__ == '__main__':
    trackMultipleObjects()
