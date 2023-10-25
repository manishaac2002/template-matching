# Python program to illustrate 
# template matching 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import imutils

# Read the main image 
img = cv2.imread('images/-_2003880-_URO-_20442_20200519_Kidney_0002.JPG')

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 


# Convert it to grayscale 
img_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 

# Read the template 
template = cv2.imread('images/scan_beanShape.JPG', 0) 

# Denoising the image and template
img_gray =cv2.bilateralFilter(img_grey,11,17,17)
template =cv2.bilateralFilter(template,11,17,17)

# loop over the scales of the image
for scale in np.linspace(0.2,0.1,20)[::-1]:
    resized = imutils.resize(template,width =int(template.shape[1]*scale))
    for angle in np.arange(0,360,15):
        rotated =imutils.rotate(resized,angle) 
# Store width and height of template in w and h 
        w, h = template.shape[::-1] 

        #  perform match operations
        res =cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

        # Specify a threshold
        threshold =0.9

        # store the coordinates of matched area in a numpy array
        loc =np.where(res >=threshold)

        if loc:
             for pt in zip(*loc[::-1]):
                  cv2.rectangle(img2, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

plt.imshow(img2)
plt.show()


