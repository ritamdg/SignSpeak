import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3  # Import Text-to-Speech library
import warnings

warnings.filterwarnings("ignore")

# --- 1. SETUP TEXT-TO-SPEECH ---
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# --- 2. LOAD MODEL ---
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: 'model.p' not found. Please run train_model.py first.")
    exit()

# --- 3. SETUP MEDIAPIPE ---
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# --- 4. VARIABLES FOR SMART TYPING ---
current_sentence = ""
last_prediction = ""
frame_counter = 0
THRESHOLD_FRAMES = 15  # How long to hold a sign before typing (adjust if too fast/slow)

print("System Ready!")
print("Controls: 'Space' -> Space | 'Backspace' -> Delete | 'S' -> Speak | 'Q' -> Quit")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret: break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make the UI look cool (Dark banner at the top)
    cv2.rectangle(frame, (0, 0), (W, 80), (20, 20, 20), -1)

    results = hands.process(frame_rgb)

    predicted_character = ""

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Visual feedback: Draw the skeleton
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]

            # --- SMART TYPING LOGIC ---
            # If the prediction matches the last frame, increase confidence counter
            if predicted_character == last_prediction:
                frame_counter += 1
            else:
                # If prediction changed, reset counter
                last_prediction = predicted_character
                frame_counter = 0

            # If we held the sign long enough, type it!
            if frame_counter == THRESHOLD_FRAMES:
                current_sentence += predicted_character
                frame_counter = 0  # Reset so it doesn't type 'AAAA' instantly

            # Draw bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    # --- DISPLAY SENTENCE ---
    # Show what has been typed so far
    cv2.putText(frame, "Sentence: " + current_sentence, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

    cv2.imshow('Smart Sign Language Detector', frame)

    # --- KEYBOARD CONTROLS ---
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
    elif key == 32:  # Space bar
        current_sentence += " "
    elif key == 8:  # Backspace
        current_sentence = current_sentence[:-1]
    elif key & 0xFF == ord('s'):  # 'S' key to speak
        if current_sentence:
            print(f"Speaking: {current_sentence}")
            engine.say(current_sentence)
            engine.runAndWait()  # This pauses the video briefly while speaking
            current_sentence = ""  # Optional: Clear sentence after speaking

cap.release()
cv2.destroyAllWindows()