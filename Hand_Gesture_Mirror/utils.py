import cv2 as cv
import mediapipe as mp
import numpy as np
from gesture import classify_pose

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def detect_and_classify(frame):
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    pose = "Unknown"
    hand_landmarks = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            pose = classify_pose(hand_landmarks)
            break
    return pose, hand_landmarks

def get_edge_hand(frame, hand_landmarks):
    h, w, _ = frame.shape
    x_vals = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_vals = [int(lm.y * h) for lm in hand_landmarks.landmark]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    margin = 30
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w, x_max + margin)
    y_max = min(h, y_max + margin)

    hand_crop = frame[y_min:y_max, x_min:x_max]
    gray = cv.cvtColor(hand_crop, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200)
    edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    resized = cv.resize(edges_bgr, (300, 300))
    return resized

def draw_box(frame):
    h, w, _ = frame.shape
    box_w, box_h = int(w * 0.6), int(h * 0.6)
    x1, y1 = (w - box_w) // 2, (h - box_h) // 2
    x2, y2 = x1 + box_w, y1 + box_h
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
