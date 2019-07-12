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

# TODO -- GOALS -- TODO
# 1. Better Template
# 3. Improve OCR
# 4. Improve Alignment
# 5. Improve Pre-Screen

#global variable declaration
GOOD_MATCH_PERCENT = .08
SRC_PATH = "/Users/ngover/Documents/TestPrograms/Images/"
IMG = "Texas/V/TX_V_test14.png"
BLUR_THRESHOLD = 15 # TODO mess around with this
DARKNESS_THRESHOLD = 50 # TODO mess with this
CHARACTER_WHITELIST = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/-"

#start of main
def main():
        # read the images
        try:
		img = cv2.imread(SRC_PATH + IMG)
	except IOError as e:
                print("({})".format(e))

	#display results
	myDoc = buildDocument(img)
	print("\n" + myDoc.__str__())
#end of main


#start of preScreen()
def preScreen(img):
	#convert image to greyscale, compute the focus measure of the image
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
def drawBoxes(img):
	# coordinates of each field are stored as a tuple in the corresponding class for the DL
	# dob
	cv2.rectangle(img, templates.TX_V.dob[0], templates.TX_V.dob[1], (0,255,0), 20)
	# Last
	cv2.rectangle(img, templates.TX_V.last[0], templates.TX_V.last[1], (0,255,0), 20)
	# First
	cv2.rectangle(img, templates.TX_V.first[0], templates.TX_V.first[1], (0,255,0), 20)
	# address
	cv2.rectangle(img, templates.TX_V.address[0], templates.TX_V.address[1], (0,255,0), 20)	
	return img
#end of drawBoxes


#start of getText
#	This function is passed an image, or ROI, from which we will extract the text. It expects the image
#	to only contain the data, because all text inside the ROI will be read and converted to a string
#	TODO -- Make this more robust... perhaps tweak this to be more dynamic so that it can solve the OK ID prob?
def getText(roi):
	# prep image w/ resize, color convert, noise reduction & threshold.
	roi = cv2.resize(roi,None,fx=1,fy=1.5,interpolation=cv2.INTER_CUBIC)
	roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
	roi = cv2.GaussianBlur(roi, (1,3), 0)
	roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	#add a white border to the image to make it easier for OCR
	roi = cv2.copyMakeBorder(roi, 500, 500, 500, 500, cv2.BORDER_CONSTANT, value=255)	  
	# create temp file for the roi, then open, read, and close.
	cv2.imwrite(SRC_PATH + "temp.jpg", roi)
	result = pytesseract.image_to_string(Image.open(SRC_PATH + "temp.jpg"), lang='eng', config=CHARACTER_WHITELIST)
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
	#removeBorder() will not modify the image if it cannot find a rectangular object.
	imgNoBorder = transform.removeBorder(img)	
	
	#prescreen
	result = preScreen(imgNoBorder)
	if result == False:
		print("Image quality too low.")
		sys.exit(0)

	#search for best template match
	template, docType = matchToTemplate(imgNoBorder)
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
		myDoc.first = getText(imReg[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]])
		#retrieve last name coordinates from template, get text from image, assign to object
		coords = getattr(templateData, "last")
		myDoc.last = getText(imReg[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]])
		#retrieve Address coordinates from template, get text from image, assign to object
		coords = getattr(templateData, "address")
		myDoc.address = getText(imReg[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]])
		#retrieve DOB coordinates from template, get text from image, assign to object
		coords = getattr(templateData, "dob")
		myDoc.dob= getText(imReg[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]])

	#TODO remove -- display for debugging/testing
	cv2.imshow("Original", imutils.resize(img, height=500))
	cv2.imshow("Template", imutils.resize(template, height=500))
	cv2.imshow("Warped", imutils.resize(drawBoxes(imReg), height=500))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	return myDoc
#end of buildDocument
	

#start of matchToTemplate
#	This function takes an input BGR image and searches through the templates folder, attempts to match a template
#	to the image, and keeps track of the score of the match (a score of 1 is best). At the end of the loop, the
#	function returns the best match it found, along with the state & orientation of the document (i.e. TX_V for
#	a vertical TX ID).
def matchToTemplate(img):
	#set to gray
	grayImg = cleanImage(img)
	bestMatch = 0

	#loop through all the templates in the TEMPLATE src folder
	for filename in os.listdir(SRC_PATH + "Templates/"):
		if filename.endswith(".png") or filename.endswith(".jpg"): #All the templates expected to be jpg/png files
			template = cv2.imread(SRC_PATH + "Templates/" + filename)			
			grayTemplate = cleanImage(template)

			#make sure sized appropriately before pass to function, if template height/width > image height/width,
			#cv2.matchTemplate will throw exception.
			imgHeight, imgWidth = grayImg.shape
			tempHeight, tempWidth = grayTemplate.shape

			if(imgHeight > tempHeight):
				grayImg = imutils.resize(grayImg, height=tempHeight - 10)
			imgHeight, imgWidth = grayImg.shape

			if(imgWidth > tempWidth):
				grayImg = imutils.resize(grayImg, width=tempWidth - 10)

			#Try to find match
			match = cv2.matchTemplate(grayImg, grayTemplate, cv2.TM_CCORR_NORMED)
			minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(match)
			print("Filename: {}, maxVal: {}, minVal: {}".format(filename, maxVal, minVal))	
			if maxVal >= bestMatch:
				bestMatch = maxVal
				bestTemplate = template
				#creates a substring from the filename up to the second underscore
				#this isolates the state and orientation of the document
				form = filename[:filename.index('_', filename.index('_') + 1)]
	return bestTemplate, form
#end of matchToTemplate
		

if __name__ == "__main__":
    main()	
