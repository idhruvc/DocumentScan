# Document Scanner README

## Description
This program was built to function as a scanner and reader which can process (currently) Texas and Oregon IDs. The program first identifies the document, using powerful image/template matching processes from the OpenCV python library. Once the program knows what kind of document it has, it will access its template for that document, align the input image to the blank template, and perform zonal OCR using Pytesseract on the regions of the image that are expected to contain text after alignment. The text retrieved from the image is then assigned to an object for processing. Currently supports JPG and PNG files, with intent to add support for PDFs soon.

## Requirements
Use the package manager **pip** to install the necessary client libraries.
```bash
pip install opencv-contrib-python
pip install imutils
pip install numpy
pip install unidecode
pip install —upgrade google-cloud-vision
```
NOTE: if you are running the Tesseract OCR branch of the project you will also need the pytesseract library.
```bash
pip install pytesseract
```


## Project Organization
The project is separated into 4 modules:

**Transform.py** - This module contains the functions that work to find the outline of the document, make sure the picture is oriented right side up and get a birds-eye-view. Accessed by ScanID when removeBackground() is called

**ScanID.py** - Acts as the driver, this module contains main and directs most of the workflow. Contains the functions that select and align the template, and when the program terminates, this module ends up with the object that was created.

**Document.py** - Class structure for the base class, Document, and its derived classes, License and SocialSecurity. Building the ID is delegated to the derived class constructors, but to build any generic document, the access point is to call the documentFromImage() function, which drives OCR and object creation calling various methods within this module.

**TemplateData.py** - This module contains the data that gets referenced when an object is created, such as coordinates to expect information after the image is aligned to the template image, how the doc owner’s name is formatted, and driver’s license state/orientation (if applicable)

Here is essentially a high-level workflow of the project:
INSERT

#### Setup and Usage
The project expects template data to be organized hierarchically, as shown:
INSERT
This makes exploring subdirectories and examining individual features of different documents and forms organized.

To run, change the SRC_PATH variable in the ScanID.py module to the absolute path to the parent directory of the /Templates folder

After this, open a command line environment. The program only takes one argument, which should be the path to the image you want to pass into the program. You can choose to include or exclude the file extension. Program has the option to leave off the file extension for convenience. Example:
```bash
python ScanID.py /your/image/path/here.png
```
```bash
python ScanID.py /your/image/path/here
```

## Help for Future Development/Troubleshooting:
To show an image at any step, insert the following cv2 methods, passing the string you want displayed on the window when the image displays itself, and the name of the image itself. Then press any key on the keyboard to close the window.
```Python
cv2.imshow(“foo”, foo)
cv2.waitKey(0)
```

Common Issues:
* OCR accuracy — problems here usually fall into two categories:
    * Tesseract
            * Tesseract OCR returned bad results on a clear image: For example, the letter ‘M’ can be mistakenly read as ‘IVI’. Resizing the image before passing to tesseract usually fixes issues like this. Tesseract results are optimized with black text against a white background at or close to 300dpi, and sometimes the calculated region resize done in the readROI() function of the Document module is not sufficient if the text is unexpectedly small.
            * Insufficient cleanup before passing image to tesseract: Sometimes tesseract can misinterpret that noise leftover in the thresholded image represents letters. The way to fix this is to adjust erosion/dilation settings and blur size/setting. Conversely, there can be TOO much noise removal, and part of the text can get left out in thresholding, or be too blurry to process correctly.
    * Google Cloud Vision
            * Images have to be resized so that they can all be stitched together on top of each other and batched to GCP in one image, rather than 5 individual ones. While this cuts down the cost, sometimes the warp can interfere with the image clarity, or prompt the API to return an incorrect response.
* Template Selection inaccuracy:
    * Template Select is very dependent on the success of background removal. Getting the license isolated and as aligned as possible is key to good results here. Template matching algorithm expects the license to be as focused as possible and right-side up.
    * This could be fixed by a more robust matching algorithm. The matching function currently loops over different scales of the template image, and records the match at each size. However, this is not as affine to rotation and perspective changes. This could be added into the current algorithm, but it would likely be very slow. The other way to fix this is to modify the match function to look at key point matches rather than laying a template over the image and recording the similarity, however it is more difficult to ‘measure’ a key point match to find the best match score.
* Template Selection Speed:
    * The reason the algorithm is somewhat slow is because it has O(N^2) complexity. The outer loop goes through all the template images in a folder, and the inner loop iterates over the image at multiple different sizes and measures the match score for each — this is more robust than one fixed size match score for each template, but has its tradeoffs.
    * A smarter way to organize this may be to create a data structure which organizes the templates in order of most to least populated states. Assuming a good picture, the common cases will be faster.
* Template match was correct, but the image did not align correctly:
    * Make sure the template that you’re trying to use is as clean and clear as possible. High resolution, sharp details and even lighting are the gold standard for matching/alignment. 
    * Try adjusting GOOD_MATCH_PERCENT in the ScanID module. I got the best results with 15%-25%, but sometimes there are individual cases where lower or higher values produce better results.
* Why does the result say my image quality was too low? 
    * Perhaps the picture is too dark, or too blurry. Try submitting a clearer picture, or lowering the values of BLUR_THRESHOLD and/or DARKNESS_THRESHOLD.
    * There exists a bug where sometimes face detection settings are too low (or too high), and sometimes a bad region selection in the background removal step receives a false-positive result in the face check, which may be a small/random region of the image, which the prescreen function will consider too low resolution and return a bad result.
