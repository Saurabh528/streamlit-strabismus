import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import math
from datetime import datetime
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import pandas as pd
import os
import queue
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import av

font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)
org1 = (50,100)

# fontScale
fontScale = 1

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 1



def get_unique(c):
    temp_list = list(c)
    temp_set = set()
    for t in temp_list:
        temp_set.add(t[0])
        temp_set.add(t[1])
    return list(temp_set)
    
mp_face_mesh = mp.solutions.face_mesh
connections_iris = mp_face_mesh.FACEMESH_IRISES
iris_indices = get_unique(connections_iris)

connections_left_eyes =  mp_face_mesh.FACEMESH_LEFT_EYE
left_eyes_indices = get_unique(connections_left_eyes)

connections_right_eyes =  mp_face_mesh.FACEMESH_RIGHT_EYE
right_eyes_indices = get_unique(connections_right_eyes)

iris_right_horzn = [469,471]
iris_right_vert = [470,472]
iris_left_horzn = [474,476]
iris_left_vert = [475,477]

def slope(x1, y1, x2, y2): # Line slope given two points:
    return (y2-y1)/(x2-x1)
def angle(s1, s2): 
    return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))
def distance(x1, y1, x2, y2):
    return (((x2 - x1)**2 +(y2 - y1)**2)**0.5)


leld = []
lerd = []
reld = []
rerd = []

pos_sim_values = []
temp_leld = 0
temp_lerd = 0
temp_reld = 0
temp_rerd = 0

last_value = None
start_timestamp = None
change_timestamp = None


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    last_value = None
    start_timestamp = None
    change_timestamp = None
    leld = []
    lerd = []
    reld = []
    rerd = []

    pos_sim_values = []
    temp_leld = 0
    temp_lerd = 0
    temp_reld = 0
    temp_rerd = 0


    st.write("We are in")
    img = frame.to_ndarray(format="bgr24")
    # You can process the image/frame here if needed
    with mp_face_mesh.FaceMesh(
        static_image_mode = True ,
        max_num_faces = 2 ,
        refine_landmarks = True ,
        min_detection_confidence = 0.5) as face_mesh:
        #flag = 0
        frame = img
        results = face_mesh.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        try:
            for face_landmark in results.multi_face_landmarks:
                lms = face_landmark.landmark
                d= {}
                for index in iris_indices:
                    x = int(lms[index].x*frame.shape[1])
                    y = int(lms[index].y*frame.shape[0])
                    d[index] = (x,y)
                black = np.zeros(frame.shape).astype("uint8")
#                 for index in iris_indices:
#                     #print(index)
#                     cv2.circle(frame,(d[index][0],d[index][1]),2,(0,255,0),-1)

                centre_right_iris_x_1 = int((d[iris_right_horzn[0]][0] + d[iris_right_horzn[1]][0])/2)
                centre_right_iris_y_1 = int((d[iris_right_horzn[0]][1] + d[iris_right_horzn[1]][1])/2)

                centre_right_iris_x_2 = int((d[iris_right_vert[0]][0] + d[iris_right_vert[1]][0])/2)
                centre_right_iris_y_2 = int((d[iris_right_vert[0]][1] + d[iris_right_vert[1]][1])/2)


                centre_left_iris_x_1 = int((d[iris_left_horzn[0]][0] + d[iris_left_horzn[1]][0])/2)
                centre_left_iris_y_1 = int((d[iris_left_horzn[0]][1] + d[iris_left_horzn[1]][1])/2)

                centre_left_iris_x_2 = int((d[iris_left_vert[0]][0] + d[iris_left_vert[1]][0])/2)
                centre_left_iris_y_2 = int((d[iris_left_vert[0]][1] + d[iris_left_vert[1]][1])/2)

                centre_left_iris_x = int((centre_left_iris_x_1 + centre_left_iris_x_2)/2)
                centre_left_iris_y = int((centre_left_iris_y_1 + centre_left_iris_y_2)/2)

                centre_right_iris_x = int((centre_right_iris_x_1 + centre_right_iris_x_2)/2)
                centre_right_iris_y = int((centre_right_iris_y_1 + centre_right_iris_y_2)/2)

                cv2.circle(frame,(centre_right_iris_x,centre_right_iris_y),2,(0,255,0),-1)
                cv2.circle(frame,(centre_left_iris_x,centre_left_iris_y),2,(0,255,0),-1)

              

                w = ((centre_right_iris_x - centre_left_iris_x)**2 + (centre_right_iris_y - centre_left_iris_y)**2)**0.5
                
                W = 6.3
                
                d = 30
                
                f = (w*d)/W
                
                df = pd.read_csv('dis_cal.csv')
                
                    
                
                if len(df)<30:
                    new_data = {"focus": f}

                    df = df.append(new_data, ignore_index=True)
                    df.to_csv('dis_cal.csv', index=False)
                    
                    
                    
                    
                

                




            return av.VideoFrame.from_ndarray(frame, format="bgr24")

        except Exception as e:
            return av.VideoFrame.from_ndarray(frame, format="bgr24")





st.title("Distance Calibration")
st.subheader("Please sit at a distance of 30 cms from the screen to do the calibration")


x = st.checkbox("I am sitting at a distance of 30 cms from the screen", value=False)
if x:
    st.write(mp.__version__)
    st.subheader("Please Don't Move we are callibrating")
    webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    async_processing=True,
)



       
    