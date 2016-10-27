# Organizing imports
from __future__ import print_function
import numpy as np
import imutils
import cv2
 
# Loading the image
img = cv2.imread("test.jpg")

# Resizing the image to a specific width (imutils needed)
img = imutils.resize(img, width=300)

# Making a copy of the image
clone = img.copy()

# Converting the image to grayscale
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# Detecting FAST keypoints in the image; Threshold = 40
detector = cv2.FastFeatureDetector_create(40)
kps = detector.detect(grayImg)
print("No.of.keypoints: {}".format(len(kps)))
 
# Looping over the keypoints
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	# Drawing the keypoint
	cv2.circle(img, (x, y), r, (0, 255, 255), 2)
 
# Displaying the images
cv2.imshow("Images", np.hstack([clone, img]))
cv2.waitKey(0)
