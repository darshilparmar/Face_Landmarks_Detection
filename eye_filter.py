from scipy.spatial import distance as dist 
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import numpy as np 
import argparse
import time
import dlib
from scipy.spatial import ConvexHull
import cv2

def eye_aspect_ratio(eye):
	#vertical distance
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	#horizonatal distance/ eye widht
	C = dist.euclidean(eye[0], eye[3])
	hull = ConvexHull(eye)
	eyeCenter = np.mean(eye[hull.vertices, :], axis=0)
	eyeCenter = eyeCenter.astype(int)

	#aspect ratio
	#ear = (A+B)/ (2.0 * C)

	return int(C), eyeCenter


def place_eye(frame,eyeCenter,eyeSize):
	eyeSize = int(eyeSize * 1.5)
	x1 = int(eyeCenter[0,0] - (eyeSize/2))  
	x2 = int(eyeCenter[0,0] + (eyeSize/2))  
	y1 = int(eyeCenter[0,1] - (eyeSize/2))  
	y2 = int(eyeCenter[0,1] + (eyeSize/2))

	h, w = frame.shape[:2]

	if(x1<0):
		x1 = 0
	if(y1<0):
		y1 = 0
	if(x2 > w):	
		x2 = w
	if(y2>h):
		y2 = h


	eyeOverlayWidth = x2-x1
	eyeOverlayHeight = y2-y1
	eyeOverlay = cv2.resize(imgEye, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
	mask = cv2.resize(orig_mask, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
	mask_inv = cv2.resize(orig_mask_inv, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
	roi = frame[y1:y2, x1:x2]
	roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
	roi_fg = cv2.bitwise_and(eyeOverlay,eyeOverlay,mask = mask)
	dst = cv2.add(roi_bg,roi_fg)
	frame[y1:y2, x1:x2] = dst


#aspec ration theshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0

print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')



(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']	
imgEye = cv2.imread('eye1.png', -1)
orig_mask = imgEye[:,:,3]
orig_mask_inv = cv2.bitwise_not(orig_mask)
imgEye = imgEye[:,:,0:3]
origEyeHeight, origEyeWidth = imgEye.shape[:2]

vs = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
	ret, frame = vs.read()
	# frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 0)

	for rect in rects:
		x = rect.left()
		y = rect.top()
		x1 = rect.right()
		y1 = rect.bottom()

		# shape = predictor(gray,rect)
		# shape = face_utils.shape_to_np(shape)
		landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()]) 


		leftEye = landmarks[lStart:lEnd]
		rightEye = landmarks[rStart:rEnd]
		leftEyeSize, leftEyeCenter = eye_aspect_ratio(leftEye)
		righEyeSize, rightEyeCenter = eye_aspect_ratio(rightEye)
		place_eye(frame,leftEyeCenter,leftEyeSize)
		place_eye(frame,rightEyeCenter,righEyeSize)
	cv2.imshow('Frame', frame)
	if(cv2.waitKey(1) == ord("q")):
		break

vs.release()
cv2.destroyAllWindows()
