
import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

model_file = pickle.load(open('./model_numbers.p', 'rb'))

model = model_file['model']

cam = cv2.VideoCapture(0)

st.title("Sign Language Assistance")
window = st.empty()
stop_btn = st.button("Stop")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}


while cam.isOpened() and not stop_btn:

    main_data = []
    x_ = []
    y_ = []

    _, img = cam.read()

    height, width, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    final_images = hands.process(img_rgb)

    if final_images.multi_hand_landmarks:

        for hand_landmarks in final_images.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                img,
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

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 4)

        cv2.putText(img, predicted_result, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    #cv2.imshow('Sign Language', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    window.image(img, channels="RGB")

    if cv2.waitKey(1) & 0xFF == ord("q") or stop_btn:
        break
    
    #key = cv2.waitKey(10)
    #if key == 27:
        #break
    
cam.release()
cv2.destroyAllWindows()
