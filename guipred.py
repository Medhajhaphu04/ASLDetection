import pickle

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk

from PIL import ImageTk, Image

model_dict = pickle.load(open('./finalmodelepoch15.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels = {0: 'A', 1: 'B', 2: 'C'}
labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
          13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
          25: 'Z'}

# Initialize the GUI
window = tk.Tk()
window.title("ASL Detection")
window.geometry("800x600")

# Create a canvas to display the video feed
canvas = tk.Canvas(window, width=800, height=500)
canvas.pack()

# Create a label to display the predicted ASL character
label = tk.Label(window, text="", font=("Arial", 48))
label.pack()


def process_video():
    # Open the video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame from the video capture
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip the frame horizontally for a mirror effect
        # frame_rgb = cv2.flip(frame_rgb, 1)

        # Process the frame with the hand detection model
        results = hands.process(frame_rgb)

        # Check if any hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Convert normalized landmarks to pixel coordinates
                mp_drawing.draw_landmarks(
                                    frame,  # image to draw
                                    hand_landmarks,  # model output
                                    mp_hands.HAND_CONNECTIONS,  # hand connections
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style())
                height, width, _ = frame.shape
                landmarks = np.zeros((21, 2), dtype=np.int32)
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    landmarks[idx] = (int(landmark.x * width), int(landmark.y * height))

                # Reshape the landmarks array
                landmarks = landmarks.reshape(-1, 42)

                # Normalize the landmarks to a range of 0 to 1
                landmarks = landmarks / 255.0

                # Make a prediction using the trained model
                prediction = model.predict(landmarks)
                predicted_class = np.argmax(prediction)

                # Get the predicted ASL character
                asl_char = labels[predicted_class]

                # Display the predicted ASL character on the label
                label.config(text=asl_char)

        # Display the frame on the canvas
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800, 500))
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img

        # Update the GUI
        window.update()

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()


# Start the video processing function
process_video()

# Start the GUI main loop
window.mainloop()

# while True:
#
#     data_aux = []
#     x_ = []
#     y_ = []
#
#     ret, frame = cap.read()
#
#     H, W, _ = frame.shape
#
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
#
#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#
#                 x_.append(x)
#                 y_.append(y)
#
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))
#
#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10
#
#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10
#
#         prediction = model.predict([np.asarray(data_aux)])
#
#         predicted_character = labels_dict[int(prediction[0])]
#
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                     cv2.LINE_AA)
#
#     cv2.imshow('frame', frame)
#     cv2.waitKey(1)
#
#
# cap.release()
# cv2.destroyAllWindows()
