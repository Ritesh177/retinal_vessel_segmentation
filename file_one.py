# -*- coding: utf-8 -*-
import cv2
import numpy as np

# read the image
img = cv2.imread(r"C:\Users\OCAC\Desktop\project_min\DRIVE\test\images\01_test.tif")
#cv2.imshow("o",img)

# convert to gray
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized = clahe.apply(gray)

# apply morphology
kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (5,5))
morph = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)


# divide gray by morphology image
division = cv2.divide(equalized, morph, scale=255)

# threshold
thresh = cv2.threshold(division, 0, 255, cv2.THRESH_OTSU )[1] 

# invert
thresh = 255 - thresh

# find contours and discard contours with small areas
mask = np.zeros_like(thresh)
contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]


area_thresh = 10000
for cntr in contours:
    area = cv2.contourArea(cntr)
    if area > area_thresh:
        cv2.drawContours(mask, [cntr], -1, 255, 2)
        
# apply mask to thresh
result1 = cv2.bitwise_and(thresh, mask)
mask = cv2.merge([mask,mask,mask])
result2 = cv2.bitwise_and(img, mask)

# save results
#cv2.imwrite('retina_eye_division.jpg',division)
#cv2.imwrite('retina_eye_thresh.jpg',thresh)
#cv2.imwrite('retina_eye_mask.jpg',mask)
#cv2.imwrite('retina_eye_result1.jpg',result1)
#cv2.imwrite('retina_eye_result2.jpg',result2)

# show results
cv2.imshow('morph', morph)  
cv2.imshow('division', division)  
cv2.imshow('thresh', thresh)  
cv2.imshow('mask', mask)  
cv2.imshow('result1', result1)  
cv2.imshow('result2', result2)
#cv2.imshow('result2', equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

