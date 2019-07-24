import imutils
import pytesseract
import os
import cv2
import sys
from PIL import Image
from unidecode import unidecode
from pathlib import Path
import TemplateData as templates
import Document as document
import Transform as transform
import numpy as np

# TODO -- GOALS -- TODO
# 1. Better Template
# 2. Improve background removal step - IN PROGRESS
# 3. Improve OCR - IN PROGRESS
# 4. Improve Alignment - NEXT
# 5. Improve Pre-Screen


#global variable declaration
GOOD_MATCH_PERCENT = .15
SRC_PATH = "/Users/ngover/Documents/TestPrograms/Images/"
IMG = "Samples/TX_V_test12.png"
BLUR_THRESHOLD = 15 # TODO mess around with this
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
		print("Image quality too low, try retaking.")
		sys.exit(0)

	#search for best template match
	template, docType = selectTemplate(imgNoBorder, background)
	#line up the input image with the selected template so that the data will be exactly where we expect
	imgAligned = alignToTemplate(img, template)
	
	if docType.startswith("SSN"): # social security card, process as such	
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
		myDoc.first = readROI(imgAligned[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]], NAME_WHITELIST)
		#retrieve last name coordinates from template, get text from image, assign to object
		coords = getattr(templateData, "last")
		myDoc.last = readROI(imgAligned[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]], NAME_WHITELIST)
		#retrieve Address coordinates from template, get text from image, assign to object
		coords = getattr(templateData, "address")
		myDoc.address = readROI(imgAligned[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]], ADDR_WHITELIST)
		#retrieve DOB coordinates from template, get text from image, assign to object
		coords = getattr(templateData, "dob")
		myDoc.dob= readROI(imgAligned[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]], NUM_WHITELIST)
		#retrieve expiration coordinates from template, get text from image, assign to object
		coords = getattr(templateData, "expiration")
		myDoc.expiration = readROI(imgAligned[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]], NUM_WHITELIST)

	#TODO remove -- display for debugging/testing
	cv2.imshow("Original", imutils.resize(img, height=500))
	cv2.imshow("Template Selection", imutils.resize(template, height=500))
	cv2.imshow("Original Aligned to Template", imutils.resize(drawBoxes(imgAligned, docType), height=500))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	return myDoc
#end of buildDocument


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

	#measure the mean darkness level for the image
	#(more true to document readability when background is successfully removed)
	light = np.mean(img)

	print("Darkness level: {}".format(light))	

	if light < DARKNESS_THRESHOLD:
		return False

	return True
#end of preScreen

	
#start of selectTemplate
#	TODO
def selectTemplate(img, background, location=SRC_PATH+"Templates/"):
	h,w = img.shape[:2]
	orientation = ""

	#check if image still contains background. If it does not, we can figure out whether the document submitted is horizontal or
	#vertical based on ratio of h:w
	if not background:
		h,w = img.shape[:2]
		if h > w: #document is vertical
			orientation = "_V"
		elif w > h: #document is horizontal
			orientation = "_H"

	#TODO remove this -- for debug
	cv2.imshow("Corrected", imutils.resize(img, height=500))
	cv2.waitKey(0)

	#Get the filename of the format that had the best match in the input image. the split() function gives the name of the file without
	#the .jpg or .png extension.
	form = multiScaleTemplateSelect(img, location).split('.')[0]

	print("Searching " + form + " directory...")	
	
	#update location to be the subdirectory containing all of the images for the specified form
	location = location + form + "/"
	#update form so that it will contain the filename and orientation if orientation was already determined
	form += orientation	
	
	#check if the file we're looking for with the specified form and orientation exists.
	if os.path.isfile(location + form + ".jpg"):
		bestTemplate = cv2.imread(location + form + ".jpg")
	elif os.path.isfile(location + form + ".png"):
		bestTemplate = cv2.imread(location + form + ".png")
	#if the first two branches of elif fail, then the file does not exist, that means one of two cases:
	#1. outline of document was not found to assign a vertical or horizontal orientation
	#2. Layout was not found (some states, such as oregon have different formats with DL picture on the left or right side.
	#3. (most unlikely) false match to the state/template occurred in the first loop/
	elif background is True:
		#go into features subdirectory, this contains all the unique features for each license type. The best match will contain
		#the form name in the filename. The form exists between the first underscore and the file extension. 
		#ex filenames: feature1_V.jpg, feature3_H2.png
		bestFeatureMatch = multiScaleTemplateSelect(img, location + "Features/")
		form = form + "_" +  bestFeatureMatch.split("_")[1].split('.')[0]

		#now that we have form name, attempt to read the CORRECT template from the /Templates/State/ folder.
		bestTemplate = None
		bestTemplate = cv2.imread(location + form + ".jpg")
		if bestTemplate is None:
			bestTemplate = cv2.imread(location + form + ".png")

	print("FORM: {}".format(form))
	return bestTemplate, form
#end of selectTemplate
		

#start of multiScaleTemplateSelect()
#	This function loops through all templates in the subdirectory, whose path is stored as a string in the variable named
#	location. The function is also passed the image itself as an openCV object. The function loops over multiple scales
#	of the input image, trying to match each template file in the subdirectory to the image. When the loop finishes, the filename
#	of the image that best matched the input image is returned in the form of a string.
def multiScaleTemplateSelect(img, location):
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	grayImg = cv2.GaussianBlur(grayImg, (5,5), 0)
	bestScore = 0
	
	#Loop through all files in the subdirectory stored in the variable location
	for filename in os.listdir(location):
		if filename.endswith(".png") or filename.endswith(".jpg"): #All the templates expected to be jpg/png files
			template = cv2.imread(location + filename)
			template = imutils.resize(template, height=30)
			grayTemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
			grayTemplate = cv2.GaussianBlur(grayTemplate, (5,5), 0)
			(tH, tW) = template.shape[:2]

			#Loop over different scales of the image
			for scale in np.linspace(0.1, .5, 15)[::-1]:
				resized = imutils.resize(grayImg, width=int(grayImg.shape[1] * scale))

				#Break if the resized image is smaller than the template.	
				if resized.shape[0] < tH or resized.shape[1] < tW:
					break

				#Edge detection
				edgedImg = cv2.Canny(resized, 0, 200)
				edgedTemplate = cv2.Canny(grayTemplate, 0, 200)
				
				#get list of matches, compare best match score to bestScore
				result = cv2.matchTemplate(edgedImg, edgedTemplate, cv2.TM_CCORR_NORMED)
				minScore,maxScore,_,_ = cv2.minMaxLoc(result)
				
				print("FILE: {}, SCORE: {}".format(filename, maxScore)) #TODO remove -- for debug

				if maxScore > bestScore:
					bestScore = maxScore
					bestMatch = filename

	print("BEST MATCH: {}, SCORE: {}".format(bestMatch, bestScore)) #TODO remove-- this was for debug
	return bestMatch
#end of multiScaleTemplateSelect


#start of alignToTemplate
#	This function uses takes an image and the template it has been matched to, identifies the areas
#	of the image that correspond, and calculates the homography (essentially the relationship between
#	two perspectives of the same image, takes into account rotation and translation) using the cv2.findHomography()
#	method. The template and the original image are assumed to be the same image related by this homography, which
#	is used to warp the perspective of the input image so that it aligns with the template. Returns the input image,
#	aligned to the template.
def alignToTemplate(img, template):
	#image prep to improve OCR and alignment
	imgClean = cleanImage(img)
	templateClean = cleanImage(template)

	# Detect image keypoints & descriptors using the BRISK algorithm
	akaze = cv2.BRISK_create()
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
#end of alignToTemplate


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
	# expiration
	coords = getattr(templateData, "expiration")
	cv2.rectangle(img, coords[0], coords[1], (0,255,0), 3)
	return img
#end of drawBoxes


#start of readROI
#	This function is passed an image regoin of interest, or ROI, from which we will extract the text. It expects the image
#	to only contain the data, because all text inside the ROI will be read and converted to a string, but only
# 	for characters included in the whitelist that gets passed to the function. The calling function should know what
#	kind of input to expect. If it is looking for a date/SSN/DL#/etc, it can pass a different whitelist than if it
#	were looking for a name.
def readROI(roi, whitelist):
	#date is one line, approximate to get it close to 300 dpi, which is optimal for OCR
	if whitelist is NUM_WHITELIST: 
		roi = imutils.resize(roi, height=55)
	#name is one line, we'll approximate to get close to 300 dpi, which is optimal for OCR.
	elif whitelist is NAME_WHITELIST:
		roi = imutils.resize(roi, height=75)
	#address is 2 lines, set height to try to get as close to 300 dpi as possible
	elif whitelist is ADDR_WHITELIST:
		roi = imutils.resize(roi, height=150)

	# prep image w/ resize, color convert, noise reduction & threshold
	roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
	roi = cv2.GaussianBlur(roi, (9,9), 0)
	roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7) 
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
	roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

	#add a white border to the image to make it easier for OCR
	roi = cv2.copyMakeBorder(roi, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)

	kernel = np.ones((3,3), np.uint8)
	roi = cv2.dilate(roi, kernel, iterations=1)
	roi = cv2.erode(roi, kernel, iterations=1)

	#TODO remove imshow -- for debug
	cv2.imshow("Text ROI", imutils.resize(roi, width=400))
	cv2.waitKey(0)
		
	# create temp file for the roi, then open, read, and close.
	cv2.imwrite(SRC_PATH + "temp.jpg", roi)
	result = pytesseract.image_to_string(Image.open(SRC_PATH + "temp.jpg"), lang='eng', config=whitelist)
	os.remove(SRC_PATH + "temp.jpg")

	#Fix to common OCR bug where tesseract reads '/' in dates as '1'.
	if whitelist is NUM_WHITELIST:
		result = result.replace(" ","").strip()
		#if the line is 10 chars long, and there are 1s where slashes expected, replace the 1s at expected
		#indeces with '/'
		if len(result) is 10 and (result[2] == '1' or result[5] == '1'):		
			s = list(result)
			s[2] = '/'
			s[5] = '/'
			result = "".join(s)

	return unidecode(result)
#end of readROI


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


#call to main
if __name__ == "__main__":
    main()	
