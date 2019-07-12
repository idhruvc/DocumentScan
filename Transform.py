import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from skimage.filters import threshold_local


def order_points(pts):
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


def findCentroid(points):
	x_list = [point[0] for point in points]
	y_list = [point[1] for point in points]
	length = len(points)
	x = sum(x_list) / length
	y = sum(y_list) / length
	return (x,y)


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
#TODO figure out if i want to do this centroid stuff vvv TODO
	#find the centroid of the quadrilateral in order to dilate quadrilateral around centroid
	centroid = findCentroid(rect)    

	#dilate each point while keeping the quadrilateral centered around the centroid
	for pt in rect:
		pt[0] = centroid[0] + 1.1 * (pt[0] - centroid[0])
		pt[1] = centroid[1] + 1.1 * (pt[1] - centroid[1]) 
	
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
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
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped


def removeBorder(image):
	ratio = image.shape[0]/500.0
	orig = image.copy()
	image = imutils.resize(image, height = 500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3,3), 1)
	#locate contours and features, this will be used to find the outline of the document
	edged = cv2.Canny(gray, 50, 300, 3)
	element = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10), (5,5))
	edged =	cv2.morphologyEx(edged, cv2.MORPH_CLOSE, element)
		
	#TODO remove -- image display for debug
	cv2.imshow("Edged", imutils.resize(edged, width=500))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
		
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse = True)[:5]	
	
	#approximate the contour with a polygon	
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#		cv2.drawContours(image, approx, -1, (0,255,0),3)
#		cv2.imshow("approx", image)
#		cv2.waitKey(0)
#		cv2.destroyAllWindows()
		#if approximated contour has 4 points, assuming the ID is the subject of the image,
		#it should be the bounding box for the ID in the original image
		if len(approx) == 4:
			screenCnt = approx
			break
	try:	
		warped = four_point_transform(orig, screenCnt.reshape(4,2) * ratio)
	except: # bounding polygon could not be found, just return original image
		warped = orig	

#	c = cnts[0]
#	copy = image.copy()
#	rect = cv2.minAreaRect(c)
#	box = cv2.boxPoints(rect)
#	box = np.int0(box)
#	cv2.drawContours(copy, [box],0,(0,0,255),2)
#	cv2.imshow("Bounding Box", imutils.resize(copy,width=500))
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()

	#TODO remove -- image display for debug
	cv2.imshow("Warped", imutils.resize(warped, width=500))
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

	return warped


#SRC = "/Users/ngover/Documents/TestPrograms/Images/Texas/V/"
##imread
#img = cv2.imread(SRC + "TX_V_test9.png")
#img = removeBorder(img)
