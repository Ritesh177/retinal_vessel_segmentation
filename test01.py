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
img = cv2.imread(r"C:\Users\OCAC\Desktop\project_min\DRIVE\training\images\23_training.tif")
mask=cv2.imread(r"C:\Users\OCAC\Desktop\retinal_vessel\training\training\mask_tif\23_training_mask.tif")
man=cv2.imread(r"C:\Users\OCAC\Desktop\retinal_vessel\training\training\manual_tif\23_manual1.tif")


cv2.imshow("Mask",mask)
cv2.imshow("o",img)

# convert to green
b, g, r = cv2.split(img)
cv2.imshow('green', g)

bmanu, gmanu, rmanu = cv2.split(man)
cv2.imshow('green Manual', gmanu)

#g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(15, 15))
equalized = clahe.apply(g)
cv2.imshow("equalized",equalized)

#im1 = cv2.blur(equalized,(5,5))
#im2 = cv2.boxFilter(equalized, -1, (10, 10), normalize=True)  
#im13 = cv2.GaussianBlur(equalized,(5,5),0) 
#im12=cv2.medianBlur(equalized,5)
im13=cv2.bilateralFilter(equalized, 50, 20, 100)

#clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(15, 15))
#equalized = clahe.apply(im13)

cv2.imshow('Bilateral_filter', im13)



##try
#thresh = cv2.adaptiveThreshold(im12,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #cv2.THRESH_BINARY,11,2)
#thresh = cv2.adaptiveThreshold(im12,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            #cv2.THRESH_BINARY,11,2)



# divide 
#division = cv2.divide( equalized,im1, scale=255)
ret, thresh = cv2.threshold(im13, 127,255, cv2.THRESH_OTSU ) 
# invert
thresh =  255-thresh
cv2.imshow('thresh', thresh)



#thresh = cv2.GaussianBlur(thresh,(3,3),cv2.BORDER_DEFAULT)
#cv2.imshow('threshnew', thresh)

#backgrou8nd subtraction
#fgbg1 = cv2.createBackgroundSubtractorMOG2()
#fgmask1 = fgbg1.apply(im1)
#fgmask1 = cv2.GaussianBlur(fgmask1,(5,5),cv2.BORDER_DEFAULT)
#cv2.imshow('MOG', fgmask1)

#add=cv2.add(mask,thresh)
#cv2.imshow("addd",add)
print("mask size",mask.shape)
print("thresh shape",thresh.shape)
print("mask")
print(mask)
print("thresh")
print(thresh)
bm,gm,rm= cv2.split(mask)
print("mask")
print(gm)
add=cv2.bitwise_and(gm,thresh)
cv2.imshow("addd",add)
print(add)
print(np.sum([add[:]]))


#####Denoising
#kernel = np.zeros((10,10),np.uint8)
#erosion = cv2.erode(thresh,kernel,iterations = 1)
#opening = cv2.morphologyEx(add, cv2.MORPH_TOPHAT, kernel)
#opening = cv2.fastNlMeansDenoisingMulti(thresh, 2, 5, None, 4, 7, 35)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (2,2))
#morph = cv2.morphologyEx(add, cv2.MORPH_DILATE, kernel)
morph = cv2.dilate(add,kernel,iterations = 1)
cv2.imshow('noise clear', morph)


#accuracy calculation
result=calculation(morph,gmanu,gm)
#print(result)
l=["TPR","FPR","Accuracy","PPV","Specificity"]
for i in range(len(result)):
    print(l[i],"=",result[i]*100)
#print(result[0]*100)

cv2.imwrite('original.jpg',img)
cv2.imwrite('green_channel.jpg',g)
cv2.imwrite('clahe_en.jpg',equalized)
cv2.imwrite('bilateral_filter.jpg',im13)
cv2.imwrite('threshold_image.jpg',thresh)
cv2.imwrite('denoise.jpg',add)
cv2.waitKey(0)
cv2.destroyAllWindows()