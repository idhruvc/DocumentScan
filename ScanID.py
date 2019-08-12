import imutils
import os
import cv2
import sys
import Document as document
import Transform as transform
import numpy as np

# TODO -- GOALS -- TODO
# 1. Better Template - check?
# 2. Improve background removal step - check?
# 3. Improve OCR - IN PROGRESS
# 4. Improve Alignment - check?
# 5. Improve Pre-Screen - next

#global variable declaration
GOOD_MATCH_PERCENT = .15
SRC_PATH = "/Users/ngover/Documents/TestPrograms/Images/"
BLUR_THRESHOLD = 36
DARKNESS_THRESHOLD = 50

#start of main
#	This function serves as a driver, first by calling the function to match the image to the best template,
#	then aligning the image to the template, then pulling the data from the ID by referencing the location
#	of the bounding boxes which are expected to contain the text we are interested in.
def main():
	fullPath = SRC_PATH + "Samples/" + sys.argv[1]
	
	#add file extension if the entry did not already have it.
	if fullPath.endswith(".png") or fullPath.endswith(".jpg") or fullPath.endswith(".jpeg"):
		img = cv2.imread(fullPath)
	else:
		if os.path.isfile(fullPath + ".png"):
			img = cv2.imread(fullPath + ".png")
		elif os.path.isfile(fullPath + ".jpg"):
			img = cv2.imread(fullPath + ".jpg")
		elif os.path.isfile(fullPath + ".jpeg"):
			img = cv2.imread(fullPath + ".jpeg")
		elif fullPath.endswith(".pdf"):
			img = None	
		elif os.path.isfile(fullPath + ".pdf"):
			img = None
		else:
			img = None
	
	if img is None:
		print("Image could not be opened.")
		sys.exit(0)

	#call to removeBorder, removes border from image and warps perspective if edges can be detected
	#removeBorder() will return false if the image contains background, or true if removeBorder()
	#was not able to locate the document, and the document still has the original background..
	imgNoBackground, background = transform.removeBackground(img)	
		
	#prescreen
	if preScreen(imgNoBackground) is False:
		print("Image quality too low, try retaking.")
		sys.exit(0)

	#search for best template match, image stored in template, the name of the template stored in docType
	template, docType = selectTemplate(imgNoBackground, background)
	#line up the input image with the selected template so that the data will be exactly where we expect
	imgAligned = alignToTemplate(img, template)
	#call to the document constructor, which will set up the object and read ROIs based on the info. passed
	myDoc = document.documentFromImage(imgAligned, docType)
	#call to object's toString() method.
	print("\n" + myDoc.__str__())
		
	#TODO remove -- display for debugging/testing
#	cv2.imshow("Original", imutils.resize(img, height=500))
#	cv2.imshow("Template Selection", imutils.resize(template, height=500))
#	cv2.imshow("Original Aligned to Template", imutils.resize(drawBoxes(imgAligned, docType), height=500))
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()

	return myDoc
#end of main


#start of preScreen()
#	This function performs a very simple prelim check on the image before template match/align is attempted. It
#	takes the result from the call to removeBorder() and measures the darkness and blur of the image. If one of
#	these values is too low, the method returns false to the calling function. Else, returns true
def preScreen(img):
	#convert image to greyscale, get the variance of laplacian distribution
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	focusMeasure = cv2.Laplacian(gray, cv2.CV_64F).var()
	
#	print("Focus measure: {}".format(focusMeasure))
	
	#check whether the value is above or beneath the threshold for blur
	if focusMeasure < BLUR_THRESHOLD:
		return False

	#measure the mean darkness level for the image
	#(more true to document readability when background is successfully removed)
	light = np.mean(img)

#	print("Darkness level: {}".format(light))	

	if light < DARKNESS_THRESHOLD:
		return False

	return True
#end of preScreen

	
#start of selectTemplate
#	This function is passed an image, the flag returned from removeBorder() and the location of the folder containing the tepmlates
#	file system. The program first checks if the background is still in the image. If there is not a background, the function
#	can tell without using any searching/matching process whether the license is vertical or horizontal based on aspect ratio.
#	Then, the function loops through all the identifiers that distinguish each category of document. After this search, the 
#	function checks whether it has enough information to identify the document it has. If it does, it returns the template and
#	the corresponding name of the template. If it does not, it performs a secondary search which looks for unique identifiers
#	for each derived form of the document, then returns the best match.
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
#	cv2.imshow("Corrected", imutils.resize(img, height=500))
#	cv2.waitKey(0)

	#Get the filename of the format that had the best match in the input image. the split() function gives the name of the file without
	#the .jpg or .png extension.
	form = multiScaleTemplateSelect(img, location, background).split('.')[0]

#	print("Searching " + form + " directory...")	
	
	#update location to be the subdirectory containing all of the images for the specified form
	location = location + form + "/"
	#update form so that it will contain the filename and orientation if orientation was already determined
	form += orientation	
	
	#check if the file we're looking for with the specified form and orientation exists.
	if os.path.isfile(location + form + ".jpg"):
		bestTemplate = cv2.imread(location + form + ".jpg")
	elif os.path.isfile(location + form + ".png"):
		bestTemplate = cv2.imread(location + form + ".png")
	#if the first two branches of elif fail, then the file does not exist, that means one of 3 cases:
	#1. outline of document was not found to assign a vertical or horizontal orientation
	#2. outline found, but there are multiple forms of the document with the determined orientation
	#3. (most unlikely) false positive match to the state/template occurred in the first loop
	elif background is True:
		#Folder will have a /Features subdirectory if there is more than one form of the doctype
		if os.path.exists(location + "Features/"):
			#go into features subdirectory, this contains all the unique features for each license type.
			#The best match will contain
			bestFeatureMatch = multiScaleTemplateSelect(img, location + "Features/", background)
			form = form + "_" +  bestFeatureMatch.split("_")[1].split('.')[0]
		#Else, the only image in the directory should be the correct form.
		else:
			for filename in os.listdir(location):
				if filename.endswith(".png") or filename.endswith (".jpg"):
					form  = filename.split(".")[0]

		#now that we have form name, attempt to read the CORRECT template from the /Templates/State/ folder.
		bestTemplate = None
		bestTemplate = cv2.imread(location + form + ".jpg")
		if bestTemplate is None:
			bestTemplate = cv2.imread(location + form + ".png")

	print("FORM: {}".format(form))
	return bestTemplate, form
#end of selectTemplate
		

#start of multiScaleTemplateSelect()
#	This function is called by selectTemplate to find and record template match scores.
#	This function loops through all templates in the subdirectory, whose path is stored as a string in the variable named
#	location. The function is also passed the image itself as an openCV object. The function loops over multiple scales
#	of the input image, trying to match each template file in the subdirectory to the image. When the loop finishes, the filename
#	of the image that best matched the input image is returned in the form of a string.
def multiScaleTemplateSelect(img, location, background):
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	grayImg = cv2.GaussianBlur(grayImg, (3,3), 0)
	
	#if background has NOT been removed yet, increase scale of the image, we will have more space to search. If the background
	#is already removed, we don't have to resize, the algorithm will run quicker.
	if grayImg.shape[0] > grayImg.shape[1]: #if height > width
		grayImg = imutils.resize(grayImg, height=500)
	elif grayImg.shape[1] > grayImg.shape[0]:
		grayImg = imutils.resize(grayImg, width=500)

	bestScore = 0
	bestMatch = None

	#Loop through all files in the subdirectory stored in the variable location
	for filename in os.listdir(location):
		if filename.endswith(".png") or filename.endswith(".jpg"): #All the templates expected to be jpg/png files
			if bestMatch is None:
				bestMatch = filename			

			template = cv2.imread(location + filename)
			template = imutils.resize(template, height=35)
			grayTemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
			grayTemplate = cv2.GaussianBlur(grayTemplate, (1,1), 0)
			(tH, tW) = template.shape[:2]

			bestStateScore = 0 # TODO remove -- debug			

			#Loops through 30 different scaled images in the range from 100% to 10% the image's size.
			for scale in np.linspace(0.1, 1.0, 30)[::-1]:
				resized = imutils.resize(grayImg, width=int(grayImg.shape[1] * scale))

				#Break if the resized image is smaller than the template.	
				if resized.shape[0] < tH or resized.shape[1] < tW:
					break

				#Edge detection
				edgedImg = cv2.Canny(resized, 50, 250)
				edgedTemplate = cv2.Canny(grayTemplate, 50, 250)
				
				#get list of matches, compare best match score to bestScore
				result = cv2.matchTemplate(edgedImg, edgedTemplate, cv2.TM_CCORR_NORMED)
				minScore,maxScore,_,_ = cv2.minMaxLoc(result)	
					
				#TODO TODO TODO remove this if block -- debug
				if maxScore > bestStateScore:
					bestStateScore = maxScore

				if maxScore > bestScore:
					bestScore = maxScore
					bestMatch = filename
					#more than a 50% match is a pretty good match. This is to save time on the search.
					if maxScore > 0.5:
#						print("BEST MATCH: {}, SCORE: {},".format(bestMatch,bestScore))
						return bestMatch
#			print("FILE: {}, SCORE: {}".format(filename, bestStateScore)) #TODO remove this

#	print("BEST MATCH: {}, SCORE: {}".format(bestMatch, bestScore)) #TODO remove-- this was for debug
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
	brisk = cv2.BRISK_create()
	imgKeypoints, imgDescriptors = brisk.detectAndCompute(imgClean, None)
	templateKeypoints, templateDescriptors = brisk.detectAndCompute(templateClean, None)
	
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
	# name
	coords = getattr(templateData, "name")
	cv2.rectangle(img, coords[0], coords[1], (0,255,0), 3)
	# address
	coords = getattr(templateData, "address")
	cv2.rectangle(img, coords[0], coords[1], (0,255,0), 3)	
	# expiration
	coords = getattr(templateData, "expiration")
	cv2.rectangle(img, coords[0], coords[1], (0,255,0), 3)
	return img
#end of drawBoxes


#start of cleanImage
#	This function optomizes an image before it is passed to one of the openCV matching/alignment
#	methods. The function first converts the image to greyscale, then performs a simple noise
#	reduction.
def cleanImage(img):
        # img prep (color, blur correction, noise removal)
	imgClean = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	imgClean = cv2.GaussianBlur(imgClean, (5,5), 0) 
	return imgClean
#end of cleanImage


#call to main
if __name__ == "__main__":
    main()	
