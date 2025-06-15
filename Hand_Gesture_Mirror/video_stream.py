import cv2 as cv
from threading import Thread
import time

class VideoStream:
    def __init__(self, url):
        self.stream = cv.VideoCapture(url)
        self.frame = None
        self.stopped = False

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if self.stream.isOpened():
                ret, frame = self.stream.read()
                if ret:
                    self.frame = frame
            time.sleep(0.01)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()
