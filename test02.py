# -*- coding: utf-8 -*-
import cv2
import numpy as np


def calculation(bw,ref_bw,bw_mask):
    TP_image=cv2.bitwise_and(ref_bw,bw)
    TP=np.sum([TP_image[:]])
    
    FN_image=cv2.bitwise_and(ref_bw,cv2.bitwise_not(bw))
    FN=np.sum([FN_image[:]])
    
    FP_image=cv2.bitwise_and(cv2.bitwise_not(ref_bw),bw)
    FP=np.sum([FP_image[:]])
    
    TN_image=cv2.bitwise_and(cv2.bitwise_not(ref_bw),cv2.bitwise_not(bw))
    TN=np.sum([TN_image[:]])
    
    accuracy=(TP+TN)/(TP+FN+FP+TN)
    TPR=TP/(TP+FN)
    FPR=FP/(FP+TN)
    PPV=TP/(TP+FP)
    specificity=TN/(TN+FP)
    
    r=[TPR,FPR,accuracy,PPV,specificity]
    return r


# read the image
img = cv2.imread(r"C:\Users\OCAC\Desktop\project_min\DRIVE\training\images\35_training.tif")
mask=cv2.imread(r"C:\Users\OCAC\Desktop\retinal_vessel\training\training\mask_tif\35_training_mask.tif")
man=cv2.imread(r"C:\Users\OCAC\Desktop\retinal_vessel\training\training\manual_tif\35_manual1.tif")

#Display of image
cv2.imshow("Original",img)
cv2.imshow("Mask",mask)

# convert to green
b, g, r = cv2.split(img)
cv2.imshow('green', g)

bmanu, gmanu, rmanu = cv2.split(man)
cv2.imshow('green Manual', gmanu)

bm,gm,rm= cv2.split(mask)


##Enhancement using CLAHE
clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(15, 15))
equalized = clahe.apply(g)
cv2.imshow("CLAHE_enhanced",equalized)

##Bilateral filter
im13=cv2.bilateralFilter(equalized, 50, 20, 100)
cv2.imshow('Bilateral_filter', im13)

##Thresolding using OTSU
ret, thresh = cv2.threshold(im13, 127,255, cv2.THRESH_OTSU ) 
thresh =  255-thresh
cv2.imshow('OTSU_thresh', thresh)

##Adition of Mask and Thresold image
add=cv2.bitwise_and(gm,thresh)
cv2.imshow("addd",add)

#Morphological Dilation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (1,1))
morph = cv2.dilate(add,kernel,iterations = 1)
cv2.imshow('MOrphological_dialation', morph)


#accuracy calculation
result=calculation(morph,gmanu,gm)
l=["TPR","FPR","Accuracy","PPV","Specificity"]
for i in range(len(result)):
    print(l[i],"=",result[i]*100)

cv2.waitKey(0)
cv2.destroyAllWindows()