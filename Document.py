import cv2
import numpy as np
import TemplateData as templates
import imutils
from unidecode import unidecode
import io
from google.cloud import vision
import os


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
		
		#Figure out how to process the full name into the FN/LN fields based
		#on how the name is formatted (order of fields, commas/not, number of lines, etc.)
		nameFormat = getattr(templateData, "nameFormat")
		#crop out each ROI by retrieving coordinates for each field from the template.
		box = getattr(templateData, "name")
		name = image[box[0][1]:box[1][1], box[0][0]:box[1][0]]
		box = getattr(templateData, "dob")
		dob = image[box[0][1]:box[1][1], box[0][0]:box[1][0]]
		box = getattr(templateData, "expiration")
		expiration = image[box[0][1]:box[1][1], box[0][0]:box[1][0]]
		box = getattr(templateData, "address")
		address = image[box[0][1]:box[1][1], box[0][0]:box[1][0]]
		#add a white border to each image
		name = cv2.copyMakeBorder(name, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(255,255,255))
		dob = cv2.copyMakeBorder(dob, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(255,255,255))
		expiration = cv2.copyMakeBorder(expiration, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(255,255,255))
		address = cv2.copyMakeBorder(address, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(255,255,255))		
		#compute width of the new image, resize each field so they stack evenly
		maxWidth = max(name.shape[1], dob.shape[1], expiration.shape[1], address.shape[1])
		name = imutils.resize(name, width=maxWidth)		
		dob = imutils.resize(dob, width=maxWidth)
		expiration = imutils.resize(expiration, width=maxWidth)
		address = imutils.resize(address, width=maxWidth)	
		#record the interval where we laid each region so that we can break up the response body.
		nameRegion = (0, name.shape[0])
		dobRegion = (nameRegion[1], nameRegion[1] + dob.shape[0])
		expirationRegion = (dobRegion[1], dobRegion[1] + expiration.shape[0])
		addressRegion = (expirationRegion[1], expirationRegion[1] + address.shape[0])	
		#concatenate each image on top of one another
		combined1 = np.concatenate((name, dob), axis=0)
		combined2 = np.concatenate((expiration, address), axis=0)
		combined = np.concatenate((combined1, combined2), axis=0)		
		#temp write to disk before requesting Cloud Vision API
		tempFile = "/Users/ngover/Documents/TestPrograms/DocumentScan/test.png"
		cv2.imwrite(tempFile, combined)
		
		cv2.imshow("Combined", imutils.resize(combined, height=700))
		cv2.waitKey(0)		
		
		#create API request
		client = vision.ImageAnnotatorClient()
		with io.open(tempFile, 'rb') as image_file:
			content = image_file.read()
			image = vision.types.Image(content=content)
		response = client.text_detection(image=image)
		labels = response.text_annotations
		os.remove(tempFile)		

		nameText1 = "" #first line of name
		nameText2 = "" #second line of name (if applicable)
		dobText = ""
		expText = ""
		addrText = ""
		#get full text body to preserve newlines
		body = labels[0].description.split("\n")
		#go through each label, map each one to a field based on location
		for label in labels[1:]:
			center, text = processLabel(label)

			#separate response into different fields based on where we know it should have been located
			if center[1] >=  nameRegion[0] and center[1] <= nameRegion[1]:
				#we expect name to always be the top image, and therefore the first 1-2 lines of the
				#response. Check which line the name appeared in, and add it to the respective string.
				if text in body[0]: #first line
					nameText1 += unidecode(text) + " "
				elif text in body[1] and nameFormat > 2: #second line
					nameText2 += unidecode(text) + " "
			elif center[1] >= dobRegion[0] and center[1] <= dobRegion[1]:
				dobText += unidecode(text)
			elif center[1] >= expirationRegion[0] and center[1] <= expirationRegion[1]:
				expText += unidecode(text)
			elif center[1] >= addressRegion[0] and center[1] <= addressRegion[1]:
				addrText += unidecode(text) + " "
		nameText = nameText1.strip() + "\n" + nameText2.strip()
		#assign text to object
		self.last, self.first = parseName(nameText.strip(), nameFormat)
		self.dob = dobText.strip()
		self.expiration = expText.strip()
		self.address = addrText.strip()
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
#end of Driver's License class	


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
		nameFormat = getattr(templateData, "nameFormat")
		#lambda to turn nameFormat into # of lines input is expected to be
		lines = lambda l: int(l>2) + 1
		#Crop out regions of interest using the template's data
		box = getattr(templateData, "ssn")
		ssn = image[box[0][1]:box[1][1], box[0][0]:box[1][0]]
		box = getattr(templateData, "name")
		name = image[box[0][1]:box[1][1], box[0][0]:box[1][0]]
		#add a border to each image
		ssn = cv2.copyMakeBorder(ssn, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(255,255,255))
		name = cv2.copyMakeBorder(ssn, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(255,255,255))
		#compute the width of the new image, resize each field so they stack evenly
		newWidth = max(ssn.shape[1], name.shape[1])
		ssn = imutils.resize(ssn, width=newWidth)
		name = imutils.resize(name, width=newWidth)
		#record the interval where we laid each image so that we can break up the response body
		ssnRegion = (0, ssn.shape[0])
		nameRegion = (ssnRegion[1], ssnRegion[1] + name.shape[0])
		#concatenate images on top of one another
		combined = np.concatenate((ssn, name), axis=0)
		#temp write to disk before requestiong cloud vision API
		tempFile = "/Users/ngover/Documents/TestPrograms/DocumentScan/test.png"
		cv2.imwrite(tempFile, combined)
		#create API request
		client = vision.ImageAnnotatorClient()
		with io.open(tempFile, 'rb') as image_file:
			content = image_file.read()
			image = vision.types.Image(content=content)
		response = client.text_detection(image=image)
		labels = response.text_annotations
		os.remove(tempFile)

		#go through all the text found in the image, assign to a label based on location
		nameText = ""
		ssnText = ""
		for label in labels[1:]:
			center, text = processLabel(label)
			#remove random hyphens
			while text.endswith('-'):
				text=text[:-1].strip()
			while "--" in text:
				text = text.replace("--", "")
			#separate response into fields based on location the API found the text in
			if center[1] >= ssnRegion[0] and center[1] <= ssnRegion[1]:
				nameText += unidecode(text)
			elif center[1] >= nameRegion[0] and center[1] <= nameRegion[1]:
				ssnText += unidecode(text)
		#assign to object
		self.ssn = ssnText.strip()
		self.name = ssnText.strip()
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


#***
#***THE REST OF THE FILE CONTAINS FUNCTIONS FOR THE CONSTRUCTORS TO HELP IN CREATING DOCUMENT OBJECTS FROM IMAGES***
#***


#start of documentFromImage
#	This function serves as a 'constructor' for all derived classes of Document, and also functions as the access 
#	point for external modules to begin the OCR process. This function is called with an image object and the 
#	string representing the document type. docType is expected to match the template's name in TemplateData.py exactly.
#	Returns either a License or SocialSecurity object, does not currently support Passport OCR.
def documentFromImage(img, docType):
	if docType.startswith("SSN"): #social security
		document = SocialSecurity(img)
	elif docType.startswith("PP"): #passport
		document = None # TODO create passport constructor
	else: #driver's license
		document = License(img, docType)
	return document
#end of documentFromImage


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
			if len(names) == 1:
				#sometimes comma misread as hyphen
				names = name.split("-")
			if len(names) == 1:
				#commas can be misread as a period
				names = name.split(".")
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


#start of processLabel
#	This function takes a response label containing a text label and the bounding coordinates from the Google Cloud
#	Vision API and returns the text it contains as well as the center of the bounding box it was located in.
def processLabel(label):
	text = label.description
	centerX = 0
	centerY = 0		
	
	#calculate the center of the polygon
	for vertex in label.bounding_poly.vertices:
		centerX += vertex.x
		centerY += vertex.y
	centerX = centerX / len(label.bounding_poly.vertices)
	centerY = centerY / len(label.bounding_poly.vertices)

	return (centerX, centerY), text
#end of processLabel
