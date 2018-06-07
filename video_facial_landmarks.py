from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import numpy as np
import cv2


print("[INFO] loading facial lankmark predictor")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

vs = VideoStream(0).start()
time.sleep(2.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width = 400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector(gray,0)

	for rect in rects:
		shape = predictor(gray,rect)
		shape = face_utils.shape_to_np(shape)
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)



		for (x,y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

	cv2.imshow("Frame", frame)
	if cv2.waitKey() == ord('q'):
		break
		
vs.stop()
cv2.destroyAllWindows()







