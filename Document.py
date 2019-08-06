import cv2
import numpy as np
import TemplateData as templates
import imutils
import pytesseract
from unidecode import unidecode

NAME_WHITELIST="--oem 0 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
NUM_WHITELIST="--oem 0 --psm 7 -c tessedit_char_whitelist=0123456789-/" #this whitelist can be passed when the input is expected to be a date
ADDR_WHITELIST="--oem 0 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890-#"


#start of Document base class
class Document:
	templateName = ''
	category = ''
#end of Document class


#start of Driver's License Class
class License(Document):
	state = ''
	orientation = ''
	dob = '0/0/0000'
	first = ''
	last = ''
	address = ''
	expiration = ''

	#Driver's License Constructor
	def __init__(self, image, docType):
		#access the template's data for thei given license format
		templateData = getattr(templates, docType)
		self.templateName = docType
		self.category = "DRIVER'S LICENSE"		
		self.orientation = getattr(templateData, "orientation")
		self.state = getattr(templateData, "state")
		
		#get coordinates of ROIS from the document template
		
		#retrieve name coordinates from template, get text from the image, process into the FN/LN fields based
		#on how the name is formatted (order of fields, commas/not, number of lines, etc.)
		box = getattr(templateData, "name")
		nameFormat = getattr(templateData, "nameFormat")
		#lambda to turn nameFormat into how many lines input is expected to be
		lines = lambda l: int(l > 2) + 1
		#add comma to OCR whitelist if the name is comma separated
		if nameFormat is 1:
			whitelist = NAME_WHITELIST + ","
		else:
			whitelist = NAME_WHITELIST

		tH = image.shape[0]
		tH_inches = getTemplateDimensionsInches(docType)[0]
	
		name = readROI(image[box[0][1]:box[1][1], box[0][0]:box[1][0]], whitelist, lines(nameFormat), tH, tH_inches)
		self.last, self.first = parseName(name, nameFormat)
		#retrieve dob coordinates from template, assign to object
		box = getattr(templateData, "dob")
		self.dob = readROI(image[box[0][1]:box[1][1], box[0][0]:box[1][0]], NUM_WHITELIST, 1, tH, tH_inches)
		#retrieve expiration coordinates from template, assign to object
		box = getattr(templateData, "expiration")
		self.expiration = readROI(image[box[0][1]:box[1][1], box[0][0]:box[1][0]], NUM_WHITELIST, 1, tH, tH_inches)
		#retrieve address coordinates from template, get text from image, assign to object
		box = getattr(templateData, "address")
		self.address = readROI(image[box[0][1]:box[1][1], box[0][0]:box[1][0]], ADDR_WHITELIST, 2, tH, tH_inches).replace('\n', ', ')
	#end of Driver's License Constructor

	#Driver's License toString()
	def __str__(self):
		temp = ''
		temp = temp + 'TYPE:\t' + self.category + '\n'
        	temp = temp + 'STATE:\t' + self.state + '\n'
		temp = temp + 'ORIEN:\t' + self.orientation + '\n'
		temp = temp + 'FIRST:\t' + self.first + '\n'
		temp = temp + 'LAST:\t' + self.last + '\n'
		temp = temp + 'DOB:\t' + self.dob + '\n'
		temp = temp + 'EXP:\t' + self.expiration + '\n'
		temp = temp + 'ADDR:\t' + self.address + '\n'
		return temp
	#end of Driver's License toString()
#end of SocialSecurity class	


#start of SocialSecurity class
class SocialSecurity(Document):
	name = ''
	ssn = ''
	
	#SSN constructor	
	def __init__(self, image):
		#access the template data for Social Security Cards
		templateData = getattr(templates, "SSN_H")
		self.templateName = "SSN_H"
		self.category = "SOCIAL SECURITY"
		self.orientation = "HORIZONTAL"
		#retrieve ssn ROI from document template, assign to object
		box = getattr(templateData, "ssn")
		self.ssn = readROI(image[box[0][1]:box[1][1], box[0][0]:box[1][0]], NUM_WHITELIST, 1)
		#retrieve name ROI from document template, assign to object
		box = getattr(templateData, "name")
		self.name = readROI(image[box[0][1]:box[1][1], box[0][0]:box[1][0]], NAME_WHITELIST, 1)
	#end of SSN constructor

	#SSN toString
	def __str__(self):
		temp = ''
		temp = temp + 'TYPE:\t' + self.category + '\n'
		temp = temp + 'NAME:\t' + self.name + '\n'
		temp = temp + 'SSN:\t' + self.ssn + '\n'
		return temp
	#end of SSN toString
#end of SocialSecurity class



#***THE REST OF THE FILE CONTAINS FUNCTIONS FOR THE CONSTRUCTORS TO HELP IN CREATING DOCUMENT OBJECTS FROM IMAGES***



#start of documentFromImage
#	This function serves as a 'constructor' for all derived classes of Document. This function is called with an image
#	object and the string representing the document type. docType is expected to match the template's name in TemplateData.py
#	exactly. Returns either a License, SocialSecurity, or Passport object.
def documentFromImage(img, docType):
	if docType.startswith("SSN"): #social security
		document = SocialSecurity(img)
	elif docType.startswith("PP"): #passport
		document = None # todo
	else: #driver's license
		document = License(img, docType)
	return document
#end of documentFromImage

#start of readROI
#	This function is passed an image region of interest, or ROI, from which we will extract the text, and the height of the
#	template image, which will be needed to calculate DPI to maximize OCR results. This method expects the image
#	to only contain the data, because all text inside the ROI will be read and converted to a string, but only
# 	for characters included in the whitelist that gets passed to the function. The calling function should know what
#	kind of input to expect. If it is looking for a date/SSN/DL#/etc, it can pass a different whitelist than if it
#	were looking for a name.
def readROI(roi, whitelist, numLines, templateHeight, templateHeightInches):
	regionBrightness = np.mean(roi)
	#adjust region brightnes based on how bright the region is
	if regionBrightness < 100:
		beta = np.array([45.0])
		cv2.add(roi, beta, roi)
	elif regionBrightness > 100 and regionBrightness < 150:
		beta = np.array([20.0])
		cv2.add(roi, beta, roi)
	elif regionBrightness > 150 and regionBrightness < 200:
		beta = np.array([-20.0])
		cv2.add(roi, beta, roi)
	elif regionBrightness > 200:
		beta = np.array([-45.0])
		cv2.add(roi, beta, roi)
	
	#Set the DPI to 300 for the image we read
	ratio = roi.shape[0] / float(templateHeight)
	ratio = templateHeightInches * ratio
	newHeight = int(ratio * 300)
		
	#optimize blur settings for OCR
	if whitelist is NUM_WHITELIST and numLines == 1:
		blurSize = (7,7)
	elif numLines == 1:
		blurSize = (13,13)
	elif numLines == 2:
		if whitelist is NAME_WHITELIST:
			newHeight = int(newHeight * 2)
		else:
			newHeight = 135
		blurSize = (13,13)
	# prep image w/ resize, color convert, noise reduction
	roi = imutils.resize(roi, height=newHeight)
	roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
	roi = cv2.GaussianBlur(roi, blurSize, 0)

	#increase contrast	
	alpha = np.array([2.1])
	cv2.multiply(roi, alpha, roi)

	roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5) 

	#dilation/erosion to pronounce features and reduce noise
	kernel = np.ones((2,2), np.uint8)
	roi = cv2.dilate(roi, kernel, iterations=1)
	roi = cv2.erode(roi, kernel, iterations=2)	
	
	#add a white border to the image to make it easier for OCR
	roi = cv2.copyMakeBorder(roi, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
	
	#TODO remove imshow -- for debug
	cv2.imshow("Text ROI", imutils.resize(roi, width=400))
	cv2.waitKey(0)
	
	# retrieve text from the image
	result = pytesseract.image_to_string(roi, config=whitelist)

	#cleanup hyphens.. usually caused by noise in the image.
	while result.endswith('-'):
		result = result[:-1].strip()
	while "--" in result: # TODO haven't tested this
		result = result.replace("--", "")
	
	#Fix to common OCR bug where tesseract reads '/' in dates as '1'.
	if whitelist is NUM_WHITELIST:
		result = result.replace(" ","").replace("\n", "").strip()
		#if the line is 10 chars long, and there are 1s where slashes expected, replace the 1s at expected
		#indeces with '/'
		if len(result) is 10 and (result[2] == '1' or result[5] == '1'):		
			s = list(result)
			s[2] = '/'
			s[5] = '/'
			result = "".join(s)
	result = result.replace(" -", "")
	
	return unidecode(result)
#end of readROI

#start of parseName
#	This function takes a string and the format the fields are in, then parses the string into
#	separate strings for first/last name (middle name is part of the first name field). Function
#	accepts two arguments: the raw string, and an integer representing nameFormat. The acceptable
#	formats (and their meanings) are as follows:
#	1: FN LN
#	2: LN, FN
#	3: LN
# 	   FN
#	4: FN
#	   LN
def parseName(name, nameFormat):
	try:
		if nameFormat == 1: #FN LN
			#If there are no delimiters, the entire string will be returned in the place of the FN.
			return "", name.strip()
		elif nameFormat == 2: #LN, FN
			names = name.split(",")
			return names[0].strip().replace(",",""), names[1].strip().replace(",","")
		elif nameFormat == 3: #LN\nFN
			names = name.split("\n")
			return names[0].strip(), names[1].strip()
		elif nameFormat == 4: #FN\nLN
			names = name.split("\n")
			return names[1].strip(), names[0].strip()
		else:
			#invalid nameFormat identifier (see templateData.py for correct formats)
			return "", ""
	except:
		return "", name.strip()
#end of parseName


#start of getTemplateDimensionsInches
#	Returns the size of a given doctype in inches. This number will be used to calculate the DPI for a doctype.
#	Only expects to be passed the type of document/name of the template (string), returns H,W
def getTemplateDimensionsInches(docType):
	if "SSN" in docType:
		return 2.5, 3.8
	else: #document is an ID
		if "_V" in docType:
			return 3.370, 2.215
		elif "_H" in docType:
			return 2.215, 3.370
	return 0,0
#end of getTemplateDimensionsInches

