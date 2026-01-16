import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
# Using a context manager for hands is cleaner and safer
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = r'C:\Users\RITAM\Downloads\asl_alphabet_train\asl_alphabet_train'

data = []
labels = []

# 1. Iterate through folders (A, B, C...)
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Skip if it's not a folder (like hidden system files)
    if not os.path.isdir(dir_path):
        continue

    print(f"Processing class: {dir_}")

    for img_path in os.listdir(dir_path)[:100]:
        img_full_path = os.path.join(dir_path, img_path)

        # 2. Read the image
        img = cv2.imread(img_full_path)

        # SAFETY CHECK: Skip if image is empty or corrupted
        if img is None:
            print(f"  [!] Skipping invalid image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # 3. Process hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                # Collect landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize (Relative to the top-left of the hand bounding box)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

# 4. Save to Pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("\nSuccess! 'data.pickle' created with processed landmarks.")