import numpy as np
import imutils
import pytesseract
import os
from PIL import Image, ImageOps
import TemplateData as templates
import Document as document
import Transform as transform
from unidecode import unidecode
import cv2
import sys
import math


# TODO -- GOALS -- TODO
# 1. Better Template
# 2. Improve background removal step
# 3. Improve OCR
# 4. Improve Alignment
# 5. Improve Pre-Screen


#global variable declaration
GOOD_MATCH_PERCENT = .1
SRC_PATH = "/Users/ngover/Documents/TestPrograms/Images/"
IMG = "Texas/V/TX_V_test15.png"
BLUR_THRESHOLD = 50 # TODO mess around with this
DARKNESS_THRESHOLD = 50 # TODO mess with this
NAME_WHITELIST="--oem 0 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_WHITELIST="--oem 0 -c tessedit_char_whitelist=0123456789-/" #this whitelist can be passed when the input is expected to be a date
ADDR_WHITELIST="--oem 0 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890-"


#start of main
def main():
        # read the images
        try:
		img = cv2.imread(SRC_PATH + IMG)
	except IOError as e:
                print("({})".format(e))

	if img is not None:
		myDoc = buildDocument(img)
		print("\n" + myDoc.__str__())
	else:
		print("Image could not be opened.")
#end of main


#start of preScreen()
#	TODO -- I'd like to make this more robust
#	This function performs a very simple prelim check on the image before template match/align is attempted. It
#	takes the result from the call to removeBorder() and measures the darkness and blur of the image. If one of
#	these values is too low, the method returns false to the calling function. Else, returns true
def preScreen(img):
	#convert image to greyscale, get the variance of laplacian distribution
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	focusMeasure = cv2.Laplacian(gray, cv2.CV_64F).var()
	
	print("Focus measure: {}".format(focusMeasure))
	
	#check whether the value is above or beneath the threshold for blur
	if focusMeasure < BLUR_THRESHOLD:
		return False

	#measure the mean darkness of the image
	light = np.mean(img)

	print("Darkness level: {}".format(light))	

	if light < DARKNESS_THRESHOLD:
		return False

	return True
#end of preScreen


#start of alignImages
#	This function uses takes an image and the template it has been matched to, identifies the areas
#	of the image that correspond, and calculates the homography (essentially the relationship between
#	two perspectives of the same image, takes into account rotation and translation) using the cv2.findHomography()
#	method. The template and the original image are assumed to be the same image related by this homography, which
#	is used to warp the perspective of the input image so that it aligns with the template. Returns the input image,
#	aligned to the template.
def alignImages(img, template):
	#image prep to improve OCR and alignment
	imgClean = cleanImage(img)
	templateClean = cleanImage(template)

	# Detect image keypoints & descriptors using the AKAZE algorithm
	akaze = cv2.AKAZE_create()
	imgKeypoints, imgDescriptors = akaze.detectAndCompute(imgClean, None)
	templateKeypoints, templateDescriptors = akaze.detectAndCompute(templateClean, None)
	
	# Match corresponding features between the images
	descriptorMatcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = descriptorMatcher.match(imgDescriptors, templateDescriptors, None)
	
	# Sort matches by score, and we only want to care about the best x% of matches.
	matches.sort(key=lambda m: m.distance, reverse=False)
	numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
	matches = matches[:numGoodMatches]
	
	# Pull the coordinates of the best matches
	imgPoints = np.zeros((len(matches), 2), dtype=np.float32)
	templatePoints = np.zeros((len(matches), 2), dtype=np.float32)
	
	for i, match in enumerate(matches):
		imgPoints[i,:] = imgKeypoints[match.queryIdx].pt
		templatePoints[i,:] = templateKeypoints[match.trainIdx].pt
	
	# find homography
	h, mask = cv2.findHomography(imgPoints, templatePoints, cv2.RANSAC)
	
	# apply homography, warping the image to appear as if it were directly below the camera.
	height, width, channels = template.shape
	imgAligned = cv2.warpPerspective(img, h, (width, height))
	return imgAligned
#end of alignImages


#start of drawBoxes
#	This function is mostly for demonstration/debugging purposes. It references the ID's template data to
#	draw boxes around the regions it is reading text from.
def drawBoxes(img, docType):
	templateData = getattr(templates, docType)
	# coordinates of each field are stored as a tuple in the corresponding class for the DL
	# dob
	coords = getattr(templateData, "dob")
	cv2.rectangle(img, coords[0], coords[1], (0,255,0), 3)
	# Last
	coords = getattr(templateData, "last")
	cv2.rectangle(img, coords[0], coords[1], (0,255,0), 3)
	# First
	coords = getattr(templateData, "first")
	cv2.rectangle(img, coords[0], coords[1], (0,255,0), 3)
	# address
	coords = getattr(templateData, "address")
	cv2.rectangle(img, coords[0], coords[1], (0,255,0), 3)	
	return img
#end of drawBoxes


#start of getText
#	This function is passed an image, or ROI, from which we will extract the text. It expects the image
#	to only contain the data, because all text inside the ROI will be read and converted to a string, but only
# 	for characters included in the whitelist that gets passed to the function. The calling function should know what
#	kind of input to expect. If it is looking for a date/SSN/DL#/etc, it can pass a different whitelist than if it
#	were looking for a name.
#	TODO -- Make this more robust... perhaps tweak this to be more dynamic so that it can solve the OK ID prob?
def getText(roi, whitelist):
	# prep image w/ resize, color convert, noise reduction & threshold.
	roi = cv2.resize(roi,None,fx=1,fy=1.5,interpolation=cv2.INTER_CUBIC)
	roi = imutils.resize(roi, height=500)
	roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
	roi = cv2.GaussianBlur(roi, (3,3), 0)
	roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

	#add a white border to the image to make it easier for OCR
	roi = cv2.copyMakeBorder(roi, 500, 500, 500, 500, cv2.BORDER_CONSTANT, value=255)
	
	# create temp file for the roi, then open, read, and close.
	cv2.imwrite(SRC_PATH + "temp.jpg", roi)
	result = pytesseract.image_to_string(Image.open(SRC_PATH + "temp.jpg"), lang='eng', config=whitelist)
	os.remove(SRC_PATH + "temp.jpg")

	return unidecode(result)
#end of getText


#start of cleanImage
#	This function optomizes an image before it is passed to one of the openCV matching/alignment
#	methods. The function first converts the image to greyscale, then performs a noise reduction thru use
#	of gaussian blur and dilation/erosion
def cleanImage(img):
	kernel = np.ones((3,3), np.uint8)
        # img prep (color, blur correction, noise removal)
	imgClean = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	imgClean = cv2.GaussianBlur(imgClean, (3,9), 15) 
	imgClean = cv2.dilate(imgClean, kernel, iterations=1)
	imgClean = cv2.erode(imgClean, kernel, iterations=1)
	return imgClean
#end of cleanImage


#start of buildDocument
#	This function serves as a driver, first by calling the function to match the image to the best template,
#	then aligning the image to the template, then pulling the data from the ID by referencing the location
#	of the bounding boxes which are expected to contain the text we are interested in.
def buildDocument(img):
	#call to removeBorder, removes border from image and warps perspective if edges can be detected
	#removeBorder() will return false if the image contains background, or true if removeBorder()
	#was not able to locate the document, and the document still has the original background..
	imgNoBorder, background = transform.removeBorder(img)	
		
	#prescreen
	result = preScreen(imgNoBorder)
	
	if result == False:
		print("Image quality too low.")
		sys.exit(0)

	#search for best template match
	template, docType = matchToTemplate(imgNoBorder, background)
	#call to alignImages, aligns image to the best template match
	imReg = alignImages(img, template)
	
	if docType.startswith("SS"): # social security card, process as such	
		# TODO
		print("Social Security Card!")
		myDoc = None
	elif docType.startswith("PP"): # document is a passport, process as such
		# TODO
		print("Passport!")
		myDoc = None
	else: # document is an ID
		myDoc = document.License()
		#access the license's data & ROIs from the TemplateData module
		templateData = getattr(templates, docType)
		myDoc.orientation = getattr(templateData, "orientation")
		myDoc.state = getattr(templateData, "state")

		#get coordinates of the ROIs from the document template
		#retrieve first name coordinates from template, get text from image, assign to object
		coords = getattr(templateData, "first")
		myDoc.first = getText(imReg[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]], NAME_WHITELIST)
		#retrieve last name coordinates from template, get text from image, assign to object
		coords = getattr(templateData, "last")
		myDoc.last = getText(imReg[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]], NAME_WHITELIST)
		#retrieve Address coordinates from template, get text from image, assign to object
		coords = getattr(templateData, "address")
		myDoc.address = getText(imReg[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]], ADDR_WHITELIST)
		#retrieve DOB coordinates from template, get text from image, assign to object
		coords = getattr(templateData, "dob")
		myDoc.dob= getText(imReg[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]], NUM_WHITELIST)

	#TODO remove -- display for debugging/testing
	cv2.imshow("Original", imutils.resize(img, height=500))
	cv2.imshow("Template", imutils.resize(template, height=500))
	cv2.imshow("Warped", imutils.resize(drawBoxes(imReg, docType), height=500))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	return myDoc
#end of buildDocument
	

#start of matchToTemplate
#	This function takes an input BGR image, and a flag paramteter that is given by the function removeBorder(). When the flag is True,
#	removeBorder modified the image by removing background. If the flag is false, removeBorder could not locate a document
#	and returned the original image.. The program loops and searches through the templates folder, attempts to match a template
#	to the image, and keeps track of the score of the match (a score of 1 is best). At the end of the loop, the
#	function returns the best match it found, along with the state & orientation of the document (i.e. TX_V for
#	a vertical TX ID).
def matchToTemplate(img, background):
	h,w = img.shape[:2]
	print("Height: {}, Width: {}".format(h,w))

	#rotate document 90 degrees if needed.
	img = detectOrientation(img)
	orientation = ""

	#check if image still contains background. If it does not, we can figure out whether the document submitted is horizontal or
	#vertical based on ratio of h:w
	if not background:
		h,w = img.shape[:2]
		print("Height: {}, Width: {}".format(h,w))
		if h > w: #document is vertical
			orientation = "_V_"
		elif w > h: #document is horizontal
			orientation = "_H_"

	#perform prelim cleanup and edging	
	grayImg = cleanImage(img)
	grayImg = cv2.Canny(grayImg, 0, 150)
	bestScore = 0

	#loop through all the templates in the TEMPLATE src folder
	for filename in os.listdir(SRC_PATH + "Templates/"):
		#if we determined the orientation before the outer loop began, discard
		#templates that are not oriented the same as the document.
		if orientation is not "":
			if not orientation in filename:
				continue
		if filename.endswith(".png") or filename.endswith(".jpg"): #All the templates expected to be jpg/png files
			template = cv2.imread(SRC_PATH + "Templates/" + filename)			
			grayTemplate = cleanImage(template)
			grayTemplate = cv2.Canny(grayTemplate, 0, 150)

			#make sure sized appropriately before pass to function, if template height/width > image height/width,
			#cv2.matchTemplate will throw exception.
			imgHeight, imgWidth = grayImg.shape
			tempHeight, tempWidth = grayTemplate.shape

			if(imgHeight > tempHeight):
				grayImg = imutils.resize(grayImg, height=tempHeight - 10)
			imgHeight, imgWidth = grayImg.shape

			if(imgWidth > tempWidth):
				grayImg = imutils.resize(grayImg, width=tempWidth - 10)

			#Try to find match using cv2.TM_SQDIFF matching algorithm
			match = cv2.matchTemplate(grayImg, grayTemplate, cv2.TM_SQDIFF)
			#grab the scores of the match object
			minScore,maxScore,_,_ = cv2.minMaxLoc(match)

			print("Filename: {}, score: {}".format(filename, minScore))	
			if minScore > bestScore:
				bestScore = minScore
				bestTemplate = template
				#creates a substring from the filename up to the second underscore
				#this isolates the state and orientation of the document
				form = filename[:filename.index('_', filename.index('_') + 1)]	

	
	print("BEST MATCH: {}".format(form)) # TODO remove this
	return bestTemplate, form
#end of matchToTemplate
		

#start of detectOrientation()
#	This function determines the orientation of the text in an image so that if needed, we can rotate the image so that the
#	text aligns horizontally. Function returns the image after alignment. #TODO -- hopefully like to make this better in the
#	future, I'd like to be able to figure out whether the document is upside down or not.
def detectOrientation(image):
	orig = image.copy()
	image = imutils.resize(image, height = 500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3,3), 5)
	#locate contours and features, this will be used to find the outline of the document
	edged = cv2.Canny(gray, 0, 150)
	element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3), (1,1))
	edged =	cv2.morphologyEx(edged, cv2.MORPH_CLOSE, element)
	copy = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
			
	#Keep track of how many lines are aligned vertically and how many horizontally
	numVert = 0
	numHoriz = 0
	
	#Get the lines in the image, to find the orientation of the text in the image
	lines = cv2.HoughLinesP(edged, 1, np.pi/180, 100, None, 20, 20)

	if lines is not None:
		for i in range(0, len(lines)):
			l = lines[i][0]
			#Get the angle of the line in degrees, mod 180 to convert all angles to range [0,180)
			#keeps negative angles out of the problem, easier to measure how far from horizontal
			angle = getAngle(l[0], l[1], l[2], l[3]) % 180
				
			#if the angle is in the range [0,15] degrees or [165,180], it will be considered horizontal.
			if (angle >= 0 and angle <= 15) or (angle >= 165 and angle <= 180):
				numHoriz+=1
			#if the angle is in the range 90 +/- 15 degrees, it will be considered vertical
			elif angle >= 75 and angle <= 105:
				numVert+=1
			#else, the angle will be assumed to be a random line and will not be counted.
		
		#at the end of measuring each angle, check whether the image is made up of predominantly horiz. or vert.
		#lines. This will tell us how text in the ID is oriented.
		if numHoriz > numVert:
			return orig
		elif numVert > numHoriz:
			return np.rot90(orig) #rotate the image 90 degrees counter clockwise to align text horiz.
		else:
			return orig
#end of detectOrientation


#start of getAngle()
#	This function expects 2 cartesian coordinate pairs (representing a line) and calculates the polar angle from
#	horizontal that relates them. Returns the angle in degrees. 
def getAngle(x1, y1, x2, y2):
	radians = math.atan2(y2-y1, x2-x1)
	degrees = math.degrees(radians)
	return degrees
#end of getAngle


#call to main
if __name__ == "__main__":
    main()	
