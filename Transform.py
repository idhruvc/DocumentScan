import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from skimage.filters import threshold_local
import sys


#start of OrderPoints
#	This function orders the points in a list such that the first entry is the top left, the second
#	is the top right, third is the bottom right, and fourth is the bottom left.
def orderPoints(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect
#end of orderPoints


#start of TransformFromPoints
#	TODO
def transformFromPoints(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	boundingBox = orderPoints(pts)
	
	(topL, topR, bottomR, bottomL) = boundingBox
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	bottomWidth = np.sqrt(((bottomR[0] - bottomL[0]) ** 2) + ((bottomR[1] - bottomL[1]) ** 2))
	topWidth = np.sqrt(((topR[0] - topL[0]) ** 2) + ((topR[1] - topL[1]) ** 2))
	maxWidth = max(int(bottomWidth), int(topWidth))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightR = np.sqrt(((topR[0] - bottomR[0]) ** 2) + ((topR[1] - bottomR[1]) ** 2))
	heightL = np.sqrt(((topL[0] - bottomL[0]) ** 2) + ((topL[1] - bottomL[1]) ** 2))
	maxHeight = max(int(heightR), int(heightL))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	transformMatrix = cv2.getPerspectiveTransform(boundingBox, dst)
	warped = cv2.warpPerspective(image, transformMatrix, (maxWidth, maxHeight))
 
	# return the warped image
	return warped
#end of transformFromPoints


#start of removeBorder()
#	This function is given an image and attempts to remove the background from the image. The function
#	uses thresholding and contour approximation to estimate a rectangular bounding box and perform a 
#	perspective warp to make the image appear as if it were taken from directly above the document.
#	Returns: warped - Image w/o background. If a document was not located, this will be the original img.
#		 background - Flag variable. True if background is still in the picture, False if it was removed.
def removeBorder(image):
	ratio = image.shape[0]/500.0
	orig = image.copy()
	image = imutils.resize(image, height = 500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3), (1,1))
	background = True
	minContourArea = 10000
	i = 0	
	size = 11 # size of the Gaussian Blur

	#loop until the background is removed, adjusting gaussian blur settings each pass, with a maximum of
	#5 passes. If the outline not found after 5 passes, original image is returned.
	while background is True and i < 5:
		edged = cv2.GaussianBlur(gray.copy(), (size, size), 0)
		edged = cv2.Canny(edged, 0, 150)
		edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, element)
		
		#TODO remove -- image display for debug
		cv2.imshow("Edged", imutils.resize(edged, height=500))
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
		#locate contours and features, this will be used to find the outline of the document
		cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		cnts = imutils.grab_contours(cnts)
		cnts = sorted(cnts, key=cv2.contourArea, reverse = True)[:5]	
	
		#Search through contours, if a contour with 4 bounding points is found, it can be assumed
		#to be the ID.
		for c in cnts:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.1 * peri, True)
			#bounding poly must be rectangular and contour area must be bigger than 10000 to avoid
			#selecting random boxes in the image.
			if len(approx) == 4 and cv2.contourArea(c) >= minContourArea:
				screenCnt = approx
				break		
		try:
			#attempt to try a warp perspective
			temp = transformFromPoints(orig, screenCnt.reshape(4,2) * ratio)
			#if perspective warp success, test to make sure we didn't just get
			#a random portion of the picture by checking if we can locate a face
			faces, rotations = findFaces(temp)
			if faces is None:
				#as long as the contour area is as big or bigger than the min area threshold,
				#we can return the result because the document doesn't have to be a driver's license.
				#just be sure to return that we did not find a face.
				if cv2.contourArea(c) >= minContourArea:	
					warped = temp
					background = False
				else:
					raise Exception("Selected region too small.")
			#make sure the image isnt JUST the person's picture
			else:
				#perform the correction for the number of rotations the findFaces() method performed
				for j in range(0, rotations):
					gray = np.rot90(gray)
					temp = np.rot90(temp)
					orig = np.rot90(orig)

				for (x,y,w,h) in faces:
					area = w * h
					#if area of the face is any bigger than 1/3 of the image, the algorithm probably
					#recognized the rectangle bounding the person's picture, and returned it thinking
					#it had found the outline of the ID..
					if area > (temp.shape[0] * temp.shape[1] / 5):
						raise Exception("Incorrect region selection.")
					else:			
						warped = temp
						background = False
		except: #adjust blur/edge detection settings and try to find outline once again.
			i += 1
			size -= 2

	#if loop terminates and outline not found, return original image
	if background is True:
		warped = orig
	
	#TODO remove -- image display for debug
	cv2.imshow("Warped", imutils.resize(warped, height=500))
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

	return warped, background
#end of removeBorder


#start of findFaces()
#	This function is passed an image, tries to detect a face, rotates 90 degrees then tries again.
#	when a face is detected, the function will immediately return the face(s) detected, as well as the
#	number of rotations to find the face. This way the calling function can self-correct the image, so that
#	if this (or other) rotation/deskew functions are called, they don't have to do so much work, but rather if
#	another corrective rotation function is called, its logic only double checks this function.
def findFaces(image):
	faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	for i in range(0,4):
		copy = image.copy()
		gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
		#Now, generate a list of rectangles for all detected faces in the image.
		faces = faceCascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=15, minSize=(30,30))
		#TODO remove the imshow
		for (x,y,w,h) in faces:
			cv2.rectangle(copy, (x,y), (x+w, y+h), (0,255,0), 2)
		cv2.imshow("Face Detection", imutils.resize(copy, height=500))
		cv2.waitKey(0)
			
		if len(faces) >= 1:
			return faces, i
		else:
			image = np.rot90(image)
	return None, 0
#end of findFaces()
