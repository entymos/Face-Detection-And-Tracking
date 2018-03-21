import dlib
import cv2
import numpy as np
import time

class Object:
	x1, y1, x2, y2 = 0,0,0,0 #x1:left, y1:top, x2:right, y2:bottom
	def __init__(self, x1, y1, x2, y2):
		self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

CAM_NUMBER = 0
MODE = 0 # 0:Face Detection, 1:Object Tracking

detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(CAM_NUMBER)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1024)
#cap.set(cv2.CAP_PROP_FPS, 30)

win_width, win_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
proc_width, proc_height = 250,180#150, 100

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1 )

objs = []
roi_hist_list = []
track_window_list = []
track_run_cnt = 0

while True:
	cameraImg = cap.read()[1]
	procImg = cv2.resize(cameraImg, (proc_width, proc_height))	

	if MODE == 0:
		print('detection')
		dets = detector(procImg, 1)	
		objs = []
		roi_hist_list = []
		track_window_list = []

		for i, d in enumerate(dets):
			roi, hsv_roi, mask, roi_hist = None, None, None, None

			obj = Object(int(d.left()*win_width/proc_width), int(d.top()*win_height/proc_height), int(d.right()*win_width/proc_width), int(d.bottom()*win_height/proc_height))
			if obj.x1 <= 0:
				obj.x1 = int(0)
			if obj.y1 <= 0:
				obj.y1 = int(0)
			if obj.x2 >= win_width:
				obj.x2 = int(win_width)
			if obj.y2 >= win_height:
				obj.y2 = int(win_height)

			track_window = (obj.x1, obj.y1, obj.x2-obj.x1, obj.y2-obj.y1)
			roi = cameraImg[obj.y1:obj.y2, obj.x1:obj.x2]
			#roi = cv2.resize(roi, (100, 70))
			hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
			#mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
			#mask = cv2.inRange(hsv_roi, np.array((0., 30.,32.)), np.array((180.,255.,255.)))
			mask = cv2.inRange(hsv_roi, np.array((0, 0,0)), np.array((255,255,255)))			
			roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
			cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

			#cv2.imshow('roi', roi)

			objs.append(obj)
			track_window_list.append(track_window)
			roi_hist_list.append(roi_hist)

		if len(dets) > 0:
			MODE = 1

		#for obj in objs:		
		#	cv2.rectangle(cameraImg,(obj.x1, obj.y1),(obj.x2, obj.y2),(0,255,0),3)
		
	if MODE == 1:
		print('tracking')
		hsv = cv2.cvtColor(cameraImg, cv2.COLOR_BGR2HSV)
		for i in range(0, len(track_window_list)):
			dst = cv2.calcBackProject([hsv],[0],roi_hist_list[i],[0,180],1)
			ret, track_window = cv2.CamShift(dst, track_window_list[i], term_crit)

			x,y,w,h = track_window
			cv2.rectangle(cameraImg, (x,y), (x+w,y+h), 255, 2)
			cv2.putText(cameraImg, 'Tracked', (x-25,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)#, cv2.CV_AA)

			#pts = cv2.boxPoints(ret)
			#pts = np.int0(pts)
			#cameraImg = cv2.polylines(cameraImg,[pts],True, 255,2)

		# 20 times tracking, 1 time detection
		track_run_cnt = track_run_cnt + 1
		if track_run_cnt > 50:
			MODE = 0
			track_run_cnt = 0

	cv2.imshow('image', cameraImg)
	key = cv2.waitKey(30) & 0xff
	if key == 27:
		break

cv2.destroyAllWindows() 
cap.release() 
