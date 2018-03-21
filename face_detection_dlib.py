import dlib
import cv2

# file open
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1024)
#cap.set(cv2.CAP_PROP_FPS, 30)

win_width, win_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
proc_width, proc_height = 150, 100

detector = dlib.get_frontal_face_detector()
while(cap.isOpened()):
	framenumber = cap.get(1)
	cameraImg = cap.read()[1]

	procImg = cv2.resize(cameraImg, (proc_width, proc_height))

	# The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more faces
	dets = detector(procImg, 2)

	for i, d in enumerate(dets):
		left, top, right, bottom = 	int(d.left()*win_width/proc_width), int(d.top()*win_height/proc_height), int(d.right()*win_width/proc_width), int(d.bottom()*win_height/proc_height)
		#top  = top - int(top/4)
		#bottom  = bottom + int(bottom/8)

		cv2.rectangle(cameraImg,(left, top),(right,bottom),(0,255,0),3)
	
	cv2.imshow('image', cameraImg)

	key = cv2.waitKey(1)
	if key == 27:
		break
	if key == ord('q'):
		break

f_metadata.close()
cv2.destroyAllWindows() 
cv2.VideoCapture(CAM_NUMBER).release() 
