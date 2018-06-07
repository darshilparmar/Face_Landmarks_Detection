from imutils import face_utils
import numpy as np 
import argparse
import imutils
import dlib
import cv2


ap = argparse.ArgumentParser()
ap.add_argument('-i','--image', required=True, help='path to input image')
args = vars(ap.parse_args())

#init dlib face detector
detector = dlib.get_frontal_face_detector()
#facial landmak predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

image = cv2.imread(args['image'])
image = imutils.resize(image,width=500)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

rects = detector(gray,1)

# FACIAL_LANDMARKS_IDXS = OrderedDict([
# 	("mouth", (48, 68)),
# 	("right_eyebrow", (17, 22)),
# 	("left_eyebrow", (22, 27)),
# 	("right_eye", (36, 42)),
# 	("left_eye", (42, 48)),
# 	("nose", (27, 35)),
# 	("jaw", (0, 17))
# ])

for(i, rect) in enumerate(rects):
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	for(name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		clone = image.copy()
		cv2.putText(clone, name,(10,30), cv2.FONT_HERSHEY_SIMPLEX,
			0.7,(0,0,255), 2)

		for (x, y) in shape[i:j]:
			cv2.circle(clone,(x,y),1,(0,0,255), -1)

		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		roi = image[y:y + h, x:x + w]
		roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
 

		cv2.imshow('ROI', roi)
		cv2.imshow('Image', clone)
		cv2.waitKey(0)

	output = face_utils.visualize_facial_landmarks(image,shape)
	cv2.imshow("Image", output)
	cv2.waitKey()




