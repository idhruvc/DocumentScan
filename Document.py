import cv2
import numpy as np
import TemplateData as templates
import imutils
import pytesseract
from unidecode import unidecode

NAME_WHITELIST="--oem 0 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
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
		name = readROI(image[box[0][1]:box[1][1], box[0][0]:box[1][0]], whitelist, lines(nameFormat))
		self.last, self.first = parseName(name, nameFormat)
		#retrieve dob coordinates from template, assign to object
		box = getattr(templateData, "dob")
		self.dob = readROI(image[box[0][1]:box[1][1], box[0][0]:box[1][0]], NUM_WHITELIST, 1)
		#retrieve expiration coordinates from template, assign to object
		box = getattr(templateData, "expiration")
		self.expiration = readROI(image[box[0][1]:box[1][1], box[0][0]:box[1][0]], NUM_WHITELIST, 1)
		#retrieve address coordinates from template, get text from image, assign to object
		box = getattr(templateData, "address")
		self.address = readROI(image[box[0][1]:box[1][1], box[0][0]:box[1][0]], ADDR_WHITELIST, 2)
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
		temp = temp + 'ADDR:\t' + self.address.replace('\n', ' ') + '\n'
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



#***THE REST OF THE FILE CONTAINS FUNCTIONS TO AID IN CREATING DOCUMENT OBJECTS FROM IMAGES***



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
#	This function is passed an image region of interest, or ROI, from which we will extract the text. It expects the image
#	to only contain the data, because all text inside the ROI will be read and converted to a string, but only
# 	for characters included in the whitelist that gets passed to the function. The calling function should know what
#	kind of input to expect. If it is looking for a date/SSN/DL#/etc, it can pass a different whitelist than if it
#	were looking for a name.
def readROI(roi, whitelist, numLines):
	#date is one line, approximate to get it close to 300 dpi, which is optimal for OCR
	if whitelist is NUM_WHITELIST and numLines == 1: 
		roi = imutils.resize(roi, height=40)
		blurSize = (9,9)
	#name is one line, we'll approximate to get close to 300 dpi, which is optimal for OCR.
	elif numLines == 1:
		roi = imutils.resize(roi, height=43)
		blurSize = (15,15)
	#2 lines, set height to try to get as close to 300 dpi as possible
	elif numLines == 2:
		roi = imutils.resize(roi, height=95)
		blurSize = (13,13)
	#else, read the image as passed.


	# prep image w/ resize, color convert, noise reduction & threshold
	roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
	roi = cv2.GaussianBlur(roi, blurSize, 0)
	roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 3) 
#	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
#	roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

	#add a white border to the image to make it easier for OCR
	roi = cv2.copyMakeBorder(roi, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)

	kernel = np.ones((3,3), np.uint8)
	roi = cv2.dilate(roi, kernel, iterations=1)
	roi = cv2.erode(roi, kernel, iterations=1)

	#TODO remove imshow -- for debug
	cv2.imshow("Text ROI", imutils.resize(roi, width=400))
	cv2.waitKey(0)
		
	# create temp file for the roi, then open, read, and close.
	result = pytesseract.image_to_string(roi, config=whitelist)

	#cleanup trailing hyphens.. usually caused by noise in the image.
	while result.endswith('-'):
		result = result[:-1].strip()

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
#	TODO
def parseName(name, nameFormat):
	try:
		if nameFormat == 1: #FN LN TODO Not sure how to know where to break up a name w/o delimiters
			return "", name.strip()
		elif nameFormat == 2: #LN, FN
			names = name.split(",")
			return names[0].strip().replace(",",""), names[1].strip().replace(",","")
		elif nameFormat == 3: #LN\nFN
			names = name.split("\n")
			return names[0].strip(), names[1].strip()
		elif nameFormat == 4:
			names = name.split("\n")
			return names[1].strip(), names[0].strip()
		else:
			#invalid nameFormat identifier (see templateData.py for correct formats)
			return "", ""
	except:
		return "", name.strip()

