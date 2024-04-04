import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

model_file = pickle.load(open('./model_alphabets.p', 'rb'))

model = model_file['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

class VideoProcessor:
		
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        main_data = []
        x_ = []
        y_ = []

        height, width, _ = frm.shape

        frm_rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

        final_images = hands.process(frm_rgb)

        if final_images.multi_hand_landmarks:

            for hand_landmarks in final_images.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frm,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in final_images.multi_hand_landmarks:

                for i in range(len(hand_landmarks.landmark)):

                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):

                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    
                    main_data.append(x - min(x_))
                    main_data.append(y - min(y_))

            x1 = int(min(x_) * width) - 10
            y1 = int(min(y_) * height) - 10

            x2 = int(max(x_) * width) - 10
            y2 = int(max(y_) * height) - 10

            prediction = model.predict([np.asarray(main_data)])

            predicted_result = labels[int(prediction[0])]

            cv2.rectangle(frm, (x1, y1), (x2, y2), (255, 255, 255), 4)

            cv2.putText(frm, predicted_result, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            
            return av.VideoFrame.from_ndarray(frm, format='bgr24')

webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
				rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
	)