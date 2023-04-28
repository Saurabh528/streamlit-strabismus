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
#import streamlit_webrtc as webrt
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
focus_df = pd.read_csv('dis_cal.csv')
focus_values = focus_df.values.tolist()
focus_values = [item for sublist in focus_values for item in sublist]
focus = int(sum(focus_values)/len(focus_values))
focus = round(focus,2)


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


def process_value(value, last_value, start_timestamp, change_timestamp):
    if value != last_value:
        if change_timestamp is None:
            change_timestamp = time.time()
        elif time.time() - change_timestamp > 2:
            end_timestamp = time.time()
            time_range = (start_timestamp, end_timestamp)
            return time_range, change_timestamp
    else:
        change_timestamp = None

    return None, change_timestamp

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


csv_filename = "report_strabisums.csv"
if os.path.isfile(csv_filename):
    df = pd.read_csv(csv_filename)
else:
    df = pd.DataFrame(columns=['Start', 'End' , 'Duration' , 'Value'])

result_queue = queue.Queue()

column_names = ['Time', 'Value', 'Positional_Similarity']

# Create an empty DataFrame
# df = pd.DataFrame(columns=column_names)

# df.to_csv('out_data.csv', index=False)


class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.detections_data = []

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
                for index in iris_indices:
                    #print(index)
                    cv2.circle(frame,(d[index][0],d[index][1]),2,(0,255,0),-1)

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

                cv2.circle(black,(centre_right_iris_x,centre_right_iris_y),2,(0,0,255),-1)
                cv2.circle(black,(centre_left_iris_x,centre_left_iris_y),2,(0,0,255),-1)
                w = ((centre_right_iris_x - centre_left_iris_x)**2 + (centre_right_iris_y - centre_left_iris_y)**2)**0.5
                
                W = 6.3
               
                screen_distance = (W*focus)/w
                screen_distance = int(screen_distance)
                
                frame = cv2.putText(img, " Distance : " + str(screen_distance), (50,150), font, fontScale, color, thickness, cv2.LINE_AA)
                
                start = datetime.now().strftime("%d/%m/%y %H:%M:%S")

                new_data = {"Time": start, "Distance": str(screen_distance)}
                
                df = pd.read_csv('screen_face_distance.csv')
                if len(df)>500:
                    df = df.iloc[250:]
                    df.reset_index(drop=True, inplace=True)
                
                df = df.append(new_data, ignore_index=True)
                df.to_csv('screen_face_distance.csv', index=False)

                e= {}
                for index in left_eyes_indices:
                    x = int(lms[index].x*frame.shape[1])
                    y = int(lms[index].y*frame.shape[0])
                    e[index] = (x,y)
                for index in left_eyes_indices:
                    #print(index)
                    cv2.circle(frame,(e[index][0],e[index][1]),2,(0,255,0),-1)
                    cv2.circle(black,(e[index][0],e[index][1]),2,(0,0,255),-1)
                    if index == 263 or index == 362:
                        cv2.line(black,(e[index][0],e[index][1]),(centre_left_iris_x,centre_left_iris_y),(0,0,255),1)
                        cv2.line(frame,(e[index][0],e[index][1]),(centre_left_iris_x,centre_left_iris_y),(0,0,255),1)
                for conn in list(connections_left_eyes):
                    cv2.line(black,(e[conn[0]][0],e[conn[0]][1]),(e[conn[1]][0],e[conn[1]][1]),(0,0,255),1)

                f= {}
                for index in right_eyes_indices:
                    x = int(lms[index].x*frame.shape[1])
                    y = int(lms[index].y*frame.shape[0])
                    f[index] = (x,y)

                for index in right_eyes_indices:
                    #print(index)
                    cv2.circle(frame,(f[index][0],f[index][1]),2,(0,255,0),-1)
                    cv2.circle(black,(f[index][0],f[index][1]),2,(0,0,255),-1)
                    if index == 33 or index == 133:
                        cv2.line(black,(f[index][0],f[index][1]),(centre_right_iris_x,centre_right_iris_y),(0,0,255),1)
                        cv2.line(frame,(f[index][0],f[index][1]),(centre_right_iris_x,centre_right_iris_y),(0,0,255),1)
                for conn in list(connections_right_eyes):
                    cv2.line(black,(f[conn[0]][0],f[conn[0]][1]),(f[conn[1]][0],f[conn[1]][1]),(0,0,255),1)
                

                left_eye_left_point_index = 263
                left_eye_right_point_index = 398
                right_eye_right_point_index = 33
                right_eye_left_point_index = 133
                df = pd.DataFrame()
                df.to_csv('working1.csv')
                #st.write("I am Screening your eyes")
                

                le_lp_d = int((((e[left_eye_left_point_index][0] - centre_left_iris_x)**2 +(e[left_eye_left_point_index][1] - centre_left_iris_y)**2)**0.5))
                le_rp_d = int((((e[left_eye_right_point_index][0] - centre_left_iris_x)**2 +(e[left_eye_right_point_index][1] - centre_left_iris_y)**2)**0.5))

                re_lp_d = int((((f[right_eye_left_point_index][0] - centre_right_iris_x)**2 +(f[right_eye_left_point_index][1] - centre_right_iris_y)**2)**0.5))
                re_rp_d = int((((f[right_eye_right_point_index][0] - centre_right_iris_x)**2 +(f[right_eye_right_point_index][1] - centre_right_iris_y)**2)**0.5))
                 
                
                #frame = cv2.putText(img, "Left Eye Left Point Distance : " + str(latest_data.values[0]), (50,150), font, fontScale, color, thickness, cv2.LINE_AA)
                print("True")
           
                df_leld = pd.read_csv('leld.csv')
                data_list_leld = []
                df_lerd = pd.read_csv('lerd.csv')
                data_list_lerd = []
                df_reld = pd.read_csv('reld.csv')
                data_list_reld = []
                df_rerd = pd.read_csv('rerd.csv')
                data_list_rerd = []
                
                
                if len(df_leld) < 30:     
                    new_data = {'vals': le_lp_d}
                    df_leld = df_leld.append(new_data, ignore_index=True)
                    df_leld.to_csv('leld.csv', index=False)
                    
                    new_data = {'vals': le_rp_d}
                    df_lerd = df_lerd.append(new_data, ignore_index=True)
                    df_lerd.to_csv('lerd.csv', index=False)
                    
                    new_data = {'vals': re_lp_d}
                    df_reld = df_reld.append(new_data, ignore_index=True)
                    df_reld.to_csv('reld.csv', index=False)
                    
                    new_data = {'vals': re_rp_d}
                    df_rerd = df_rerd.append(new_data, ignore_index=True)
                    df_rerd.to_csv('rerd.csv', index=False)

                    
                    
                else:
   
                    data_list_leld = df_leld.values.tolist()
                    empty_df = pd.DataFrame(columns=df_leld.columns)
                    empty_df.to_csv('leld.csv', index=False)

                    data_list_lerd = df_lerd.values.tolist()
                    empty_df = pd.DataFrame(columns=df_lerd.columns)
                    empty_df.to_csv('lerd.csv', index=False)
                    
                    data_list_reld = df_reld.values.tolist()
                    empty_df = pd.DataFrame(columns=df_reld.columns)
                    empty_df.to_csv('reld.csv', index=False)
                    
                       
                    data_list_rerd = df_rerd.values.tolist()
                    empty_df = pd.DataFrame(columns=df_rerd.columns)
                    empty_df.to_csv('rerd.csv', index=False)

                    
                    
                    
#                 frame = cv2.putText(img, "Left Eye Left Point Distance : " + str(len(data_list_leld)), (50,150), font, fontScale, color, thickness, cv2.LINE_AA)  
#                 frame = cv2.putText(img, "right Eye right Point Distance : " + str(len(data_list_reld)), (50,200), font, fontScale, color, thickness, cv2.LINE_AA)
#                 frame = cv2.putText(img, "right Eye Left Point Distance : " + str(len(data_list_lerd)), (50,250), font, fontScale, color, thickness, cv2.LINE_AA)
#                 frame = cv2.putText(img, "Left Eye right Point Distance : " + str(len(data_list_rerd)), (50,300), font, fontScale, color, thickness, cv2.LINE_AA)
                    
                

                if len(data_list_leld) == 30:
                    

                    
#                     leld = [item for sublist in data_list_leld for item in sublist]
#                     #lerd = data_list_lerd
#                     lerd = [item for sublist in data_list_lerd for item in sublist]
#                     #reld = data_list_reld
#                     reld = [item for sublist in data_list_reld for item in sublist]
#                     #rerd = data_list_rerd
#                     rerd = [item for sublist in data_list_rerd for item in sublist]
            
                    data_list_leld = [item for sublist in data_list_leld for item in sublist]
                    temp_leld = int(sum(data_list_leld)/len(data_list_leld))

                    data_list_reld = [item for sublist in data_list_reld for item in sublist]
                    temp_reld = int(sum(data_list_reld)/len(data_list_reld))
                    

                    data_list_lerd = [item for sublist in data_list_lerd for item in sublist]
                    temp_lerd = int(sum(data_list_lerd)/len(data_list_lerd))
                    data_list_rerd = [item for sublist in data_list_rerd for item in sublist]
                    temp_rerd = int(sum(data_list_rerd)/len(data_list_rerd))

                    L2 = temp_leld
                    L1 = temp_lerd
                    R1 = temp_reld
                    R2 = temp_rerd

                    pos_sim = max((R1/R2),(L1/L2))/min((R1/R2),(L1/L2))
                    pos_sim_values.append(pos_sim)

                    value = "Normal"
                    if pos_sim > 1.42:
                        value = "Strabismus"
                    else:
                        value = "Normal"
                    
                 

                    start = datetime.now().strftime("%d/%m/%y %H:%M:%S")

                    new_data = {"Time": start, "Value": str(value),"Positional_Similarity":str(round(pos_sim,2)),
                               "LeLd":L2,"LeRd":L1,"ReLd":R1,"ReRd":R2}
                    
                    df = pd.read_csv('out_data.csv')
           
                    #df = df.append(data, ignore_index=True)
                      
                    df = df.append(new_data, ignore_index=True)
                    df.to_csv('out_data.csv', index=False)




            return av.VideoFrame.from_ndarray(frame, format="bgr24")

        except Exception as e:
            return av.VideoFrame.from_ndarray(frame, format="bgr24")


def custom_date_parser(date_string):
    return pd.to_datetime(date_string, format="%d/%m/%y %H:%M:%S")

st.title("Strabismus Screening")

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.subheader("Please remove Anaglyph glasses and remain in good lightning condtions")
    
    if st.checkbox("Start Streaming", value=True):
        column_names = ['Time','Value','Positional_Similarity','LeLd','LeRd','ReLd','ReRd']

        df = pd.DataFrame(columns=column_names)
        df.to_csv('out_data.csv', index=False)
else:
        
    x = st.checkbox("Generate Report", value=False)
    
    if x:
        #st.write(webrt.__version__)
        csv_file = 'out_data.csv'
        df = pd.read_csv(csv_file, parse_dates=['Time'], date_parser=custom_date_parser)

        # Create bins for the 'Positional_Similarity' column with 0.25 intervals
        bins = [i * 0.25 for i in range(41)]  # 41 since we want to include 10 (0.25 * 40 = 10)
        labels = [f'{i * 0.25}-{(i + 1) * 0.25}' for i in range(40)]  # 40 intervals in total
        df['pos_similarity_interval'] = pd.cut(df['Positional_Similarity'], bins=bins, labels=labels)

        # Calculate the time duration for each row
        df['duration'] = df['Time'].diff()

        # Group the DataFrame by the binned 'pos_similarity_interval' column
        grouped_df = df.groupby('pos_similarity_interval')['duration'].sum().reset_index()

        # Convert the time intervals to seconds and remove bins with no or None values
        grouped_df['duration_seconds'] = grouped_df['duration'].dt.total_seconds()
        grouped_df.dropna(subset=['duration_seconds'], inplace=True)
        grouped_df.reset_index(drop=True, inplace=True)


        # Print the result
        st.table(grouped_df[['pos_similarity_interval', 'duration_seconds']])
        
        with st.beta_container():
            col1, col2 = st.beta_columns(2)

            with col1:
                st.subheader("LELD Plot")
                st.line_chart(df['LeLd'])

            with col2:
                st.subheader("RELD Plot")
                st.line_chart(df['ReLd'])

        with st.beta_container():
            col3, col4 = st.beta_columns(2)

            with col3:
                st.subheader("LERD Plot")
                st.line_chart(df['LeRd'])

            # Plot the 'rerd' column as a line chart in the second column with a title
            with col4:
                st.subheader("RERD Plot")
                st.line_chart(df['ReRd'])


