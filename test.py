# -*- coding: utf-8 -*-
import cv2
import numpy as np
import scipy.ndimage

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def gabor_wavelet(rows, cols, kmax, f, orientation, scale, delt2):

    k = (kmax / (f ** scale)) * np.exp(1j * orientation * np.pi / 8)
    kn2 = np.abs(k) ** 2

    gw = np.zeros((rows, cols), np.complex128)

    for m in range(int(-rows/2) + 1, int(rows / 2) + 1):
        for n in range(int(-cols/2) + 1, int(cols / 2) + 1):
            t1 = np.exp(-0.5 * kn2 * (m**2 + n**2) / delt2)
            t2 = np.exp(1j * (np.real(k) * m + np.imag(k) * n))
            t3 = np.exp(-0.5 * delt2)
            gw[int(m + rows/2 - 1),int(n + cols/2 - 1)] = (kn2 / delt2) * t1 * (t2 - t3)

    return gw



img = cv2.imread(r"C:\Users\OCAC\Desktop\project_min\DRIVE\test\images\01_test.tif")

#convert to green channel
b, g, r = cv2.split(img)
cv2.imshow('green', g)

#apply equalized
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized = clahe.apply(g)
cv2.imshow('equalized', equalized)

#apply morphology
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (5,5))
#morph = cv2.morphologyEx(g, cv2.MORPH_DILATE, kernel)
#cv2.imshow('morph', morph)

img_in_gr = cv2.bitwise_not(g)
#cv2.imshow('complement', img_in_gr)

img_in_gr=im2double(img_in_gr)
#cv2.imshow('double', img_in_gr)

#Parameter setting
width = 45;
height = 45;
kmax = np.pi / 2;
f = np.sqrt( 2 );
delta = (np.pi)/3 ;
PI=3/4;
#Gabour Filtering
a=g.shape[0]
b=g.shape[1]
img_out = np.zeros((a,b),np.uint8)
cv2.imshow('img_out',img_out)
for u in range(7):
    gw=gabor_wavelet(width, height, kmax, f, u, 2, delta)
    img_out=scipy.ndimage.correlate(g, gw, mode='wrap')
cv2.imshow('img_out',img_out)
#cv2.imshow('original', img)
#print(img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

