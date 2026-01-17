import streamlit as st
import cv2
import pickle
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
from deep_translator import GoogleTranslator

# --- 1. SETUP & SESSION STATE ---
st.set_page_config(layout="wide")
st.title("SignSpeak: Smart Sign Language Detector")

if 'current_sentence' not in st.session_state:
    st.session_state['current_sentence'] = ""
if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = "None"
if 'frame_counter' not in st.session_state:
    st.session_state['frame_counter'] = 0

# PERFORMANCE FIX: Add variables to track translation
if 'last_translated_text' not in st.session_state:
    st.session_state['last_translated_text'] = ""
if 'cached_translation' not in st.session_state:
    st.session_state['cached_translation'] = ""


# --- 2. HELPER FUNCTIONS ---
def text_to_speech(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        pass


# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
frames_to_lock = st.sidebar.slider("Frames to Lock Letter", 5, 30, 15)

st.sidebar.markdown("---")
st.sidebar.subheader("Localization")
lang_options = {'English': 'en', 'Hindi': 'hi', 'Bengali': 'bn', 'Spanish': 'es'}
selected_lang_name = st.sidebar.selectbox("Translate Output To:", list(lang_options.keys()))
target_lang_code = lang_options[selected_lang_name]

stop_button = st.sidebar.button("Stop Camera", type="primary")

# --- 4. LOAD RESOURCES ---
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    st.sidebar.success(f"Model Active")
except Exception as e:
    st.error(f"Error loading model.p: {e}")
    st.stop()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# --- 5. UI LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    frame_placeholder = st.empty()

with col2:
    st.subheader("Detected Sentence (English)")
    sentence_placeholder = st.empty()

    st.markdown("---")

    st.subheader(f"Translation ({selected_lang_name})")
    translation_placeholder = st.empty()

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Space"):
            st.session_state['current_sentence'] += " "
    with b2:
        if st.button("Backspace"):
            st.session_state['current_sentence'] = st.session_state['current_sentence'][:-1]
    with b3:
        if st.button("Clear"):
            st.session_state['current_sentence'] = ""
            st.session_state['cached_translation'] = ""  # Clear translation too

    if st.button("🔊 Speak", use_container_width=True):
        if st.session_state['current_sentence']:
            t = threading.Thread(target=text_to_speech, args=(st.session_state['current_sentence'],))
            t.start()

# --- 6. MAIN LOOP ---
cap = cv2.VideoCapture(0)

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    predicted_char = "No Hand"
    prob = 0.0

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        data_aux = []
        x_ = []
        y_ = []

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
            try:
                input_data = [np.asarray(data_aux)]
                predicted_char = model.predict(input_data)[0]
                probs = model.predict_proba(input_data)
                prob = np.max(probs)

                if prob > confidence_threshold:
                    if predicted_char == st.session_state['last_prediction']:
                        st.session_state['frame_counter'] += 1
                    else:
                        st.session_state['last_prediction'] = predicted_char
                        st.session_state['frame_counter'] = 0

                    if st.session_state['frame_counter'] >= frames_to_lock:
                        st.session_state['current_sentence'] += predicted_char
                        st.session_state['frame_counter'] = 0
                        st.session_state['last_prediction'] = "None"

                        # Draw Box
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

            except Exception as e:
                pass

    # Visual Feedback
    bar_width = int(prob * 200)
    color = (0, 255, 0) if prob > confidence_threshold else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (W, 80), (20, 20, 20), -1)
    cv2.rectangle(frame, (20, 45), (20 + bar_width, 65), color, -1)
    cv2.putText(frame, f"Pred: {predicted_char} ({int(prob * 100)}%)", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- UPDATED TRANSLATION LOGIC (THE FIX) ---
    current_text = st.session_state['current_sentence']

    # Only translate if text exists AND it has CHANGED since the last frame
    if current_text and target_lang_code != 'en':
        if current_text != st.session_state['last_translated_text']:
            try:
                # This only runs ONCE per new letter
                translation = GoogleTranslator(source='auto', target=target_lang_code).translate(current_text)
                st.session_state['cached_translation'] = translation
                st.session_state['last_translated_text'] = current_text
            except:
                pass

    # Display Output
    sentence_placeholder.markdown(f"### {st.session_state['current_sentence']}")
    translation_placeholder.markdown(f"### {st.session_state['cached_translation']}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

cap.release()