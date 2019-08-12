import cv2
import Transform as transform
import ScanID as scan
import sys
from pathlib import Path
import os
import imutils

SRC_PATH = "/Users/ngover/Documents/TestPrograms/Images/"
fullPath = SRC_PATH + "Samples/" + sys.argv[1]

if os.path.isfile(fullPath + ".png"):
	image = cv2.imread(fullPath + ".png")
elif os.path.isfile(fullPath + ".jpg"):
	image = cv2.imread(fullPath + ".jpg")
elif os.path.isfile(fullPath + ".pdf"):
	print("PDF, Failed to open.")
	image = None	
else:
	image = None

if image is None:
	print("Failed to open.")
	sys.exit(0)	

noBorder, background = transform.removeBackground(image)
template, form = scan.selectTemplate(noBorder, background, location=SRC_PATH+"Templates/")
aligned = scan.alignToTemplate(image, template)
aligned = scan.drawBoxes(aligned, form)

cv2.imshow("Aligned", imutils.resize(noBorder, height=500))
cv2.waitKey(0)
cv2.deleteAllWindows()
