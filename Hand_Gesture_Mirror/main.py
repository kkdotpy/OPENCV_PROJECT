import cv2 as cv
import numpy as np
from video_stream import VideoStream
from utils import detect_and_classify, get_edge_hand, draw_box

vs1 = VideoStream('URL1').start()
vs2 = VideoStream('URL2').start()
#vs1 = VideoStream(1).start()

while True:
    frame1 = vs1.read()
    frame2 = vs2.read()

    if frame1 is None or frame2 is None:
        continue

    frame2 = cv.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    frame11 = frame1.copy()
    frame22 = frame2.copy()

    draw_box(frame1)
    draw_box(frame2)

    gesture1, landmarks1 = detect_and_classify(frame1)
    gesture2, landmarks2 = detect_and_classify(frame2)

    cv.putText(frame1, f'P1: {gesture1}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame2, f'P2: {gesture2}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    match_text = "MATCH!" if gesture1 == gesture2 and "Unknown" not in gesture1 else "NO MATCH"
    match_color = (0, 255, 0) if match_text == "MATCH!" else (0, 0, 255)

    edge1 = get_edge_hand(frame11, landmarks1) if landmarks1 else np.zeros((300, 300, 3), dtype=np.uint8)
    edge2 = get_edge_hand(frame22, landmarks2) if landmarks2 else np.zeros((300, 300, 3), dtype=np.uint8)

    combined_edge = np.hstack((edge1, edge2))
    cv.putText(combined_edge, match_text, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, match_color, 3)

    cv.imshow("Player 1", frame1)
    cv.imshow("Player 2", frame2)
    cv.imshow("Edge Detection View", combined_edge)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vs1.stop()
vs2.stop()
cv.destroyAllWindows()
