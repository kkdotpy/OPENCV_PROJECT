import mediapipe as mp

mp_hands = mp.solutions.hands

def classify_pose(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    index_extended = index_tip.y < index_mcp.y
    middle_extended = middle_tip.y < middle_mcp.y
    thumb_extended = thumb_tip.y < index_mcp.y

    if index_extended and middle_extended and not thumb_extended:
        return "VICTORY"
    elif not index_extended and not middle_extended and not thumb_extended:
        return "FIST"
    elif not index_extended and not middle_extended and thumb_extended:
        return "THUMBS_UP"
    elif index_extended and middle_extended and thumb_extended:
        return "Open"
    else:
        return "Unknown"
