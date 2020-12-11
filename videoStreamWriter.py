###############################################################################
# (c) 2005-2020 Copyright, Real-Time Innovations.  All rights reserved.       #
# No duplications, whole or partial, manual or electronic, may be made        #
# without express written permission.  Any such copies, or revisions thereof, #
# must display this notice unaltered.                                         #
# This code contains trade secrets of Real-Time Innovations, Inc.             #
###############################################################################

from time import sleep
import time

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# THIS MODULE IS THE ONE RESPONSIBLE FOR SENDING OUT THE VIDEO STREAM.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Updating the system path is not required if you have pip-installed
# rticonnextdds-connector
from sys import path as sys_path
from os import path as os_path
file_path = os_path.dirname(os_path.realpath(__file__))
sys_path.append(file_path + "/../../../")

import rticonnextdds_connector as rti
import cv2
import base64

WIDTH = 1280
HEIGHT = 720

with rti.open_connector(
        config_name="MyParticipantLibrary::MyPubParticipant",
        url=file_path + "/ImageType.xml") as connector:

    output = connector.get_output("MyPublisher::MyImageWriter")
    
    output.wait_for_subscriptions()
    
    fpsLimit = 0.25
    start_time = time.time()
    cap = cv2.VideoCapture('../video.h264')
    while cap.isOpened():
        now_time = time.time()
        if (now_time - start_time) > fpsLimit:
            (ret, frame) = cap.read()
            frame = cv2.resize(frame,(WIDTH,HEIGHT))
            _, im_arr = cv2.imencode('.jpg', frame)
            im_bytes = im_arr.tostring()
            im_b64 = base64.b64encode(im_bytes)
            output.instance.set_string("data",im_b64.decode("utf-8"))
            output.instance.set_number("videoFps",cap.get(cv2.CAP_PROP_FPS))
            print("Writing...")
            output.write()
            output.wait()
            if cv2.waitKey(0) == 27:
                break
            start_time = time.time()   
    print("Exiting...")
    cap.release()
  
