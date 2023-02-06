# -*- coding: utf-8 -*-
"""
Created on Fri May 18 18:36:59 2018

@author: Admin
"""
# to import image on to the python using opencv
import cv2 as cv
import numpy as np
image_black=np.zeros((256,256))
image_1=np.ones((256,256,3))*255
cv.imshow("black", image_black)
cv.waitKey(0)


kf_image=cv.imread(r"Kingfisher.jpg")
cv.imshow("image", kf_image)
cv.waitKey(0)

import matplotlib.pyplot as plt
plt.imshow(kf_image)

# lets load another images 
tyre=cv.imread("tyer.png")
cv.imshow("tyre", tyre)
cv.waitKey(0)
# crop1=kf_image[0:250,0:250,0]
# image_black[0:250,0:250]=crop1
# black_tyre=tyre[100:,100:,0]
# tyre[100:,100:,0]=image_1[:,:,0]
# black_tyre[100:356,100:356]=image_1[:,:,0]
#the size of the image is too big lets reduce it 
dim=(1024,786)
resized_tyre = cv.resize(tyre,dim, interpolation = cv.INTER_AREA)
cv.imshow("tyre", resized_tyre)
cv.waitKey(0)
# their are few unwanted object in images lets crop them 
tyredup=resized_tyre
crop_img = tyredup[250:586, 400:824]
cv.imshow("tyre", crop_img)
cv.waitKey(0)
# Saving an Image to Disk using Python and OpenCV
cv.imwrite("crop_img.png", crop_img)
#Image Blurring This is done by convolving the image with a normalized box filter. 
#It simply takes the average of all the pixels under kernel area and replaces the central element with this average.
blur = cv.blur(resized_tyre,(3,3))
blur = cv.blur(resized_tyre,(7,7))
cv.imshow("tyre", blur)
cv.waitKey(0)
# Image Thresholding
img = cv.imread('crop_img.png',0)
#Global Thresholding
ret,th1 = cv.threshold(img,125,255,cv.THRESH_BINARY)
cv.imshow("tyre", th1)
cv.waitKey(0)
#AVG and gaussian thersholding 
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
cv.imshow("tyre", th3)
cv.waitKey(0)
#Image Gradients
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
cv.imshow("tyre", sobely)
cv.waitKey(0)
#Canny Edge Detection in OpenCV
edges = cv.Canny(crop_img,50,150)
cv.imshow("tyre", edges)
cv.waitKey(0)
#PLT
edges_50 = cv.Canny(crop_img,50,100)
edges_25 = cv.Canny(crop_img,25,100)
plt.imshow(edges_50,cmap='gray')
cv.imshow("tyre", edges_50)
cv.waitKey(0)

#Morphological Transformations
#1) Erosion
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(resized_tyre,kernel,iterations = 1)
cv.imshow("tyre", erosion)
cv.waitKey(0)
# Contours
imges = cv.imread('crop_img.png')
imgray = cv.cvtColor(imges,cv.COLOR_BGR2GRAY)
gray = cv.bilateralFilter(imgray, 11, 17, 17)
edged = cv.Canny(gray, 20, 90)
cv.imshow("tyre", edged)
cv.waitKey(0)
contours, hierarchy = cv.findContours(edged, cv.RETR_TREE, 
                                             cv.CHAIN_APPROX_SIMPLE)
x,y,w,h = cv.boundingRect(contours[0])

for i in range(len(contours)):
    area = cv.contourArea(contours[i])
    x,y,w,h = cv.boundingRect(contours[i])
    imgrect = cv.rectangle(imges,(x,y),(x+w,y+h),(0,255,0),2)
    outfile = ('%s.jpg' % i)
    cv.imwrite(outfile, imgrect)
cv.imshow("tyre", imgrect)
cv.waitKey(0)
cv.destroyAllWindows()
#to draw circles 
(x1,y1),rad = cv.minEnclosingCircle(contours[70])
cen = (int(x1),int(y1))
radi = int(rad)
imgcir = cv.circle(crop_img,cen,radi,(0,255,0),2)
cv.imshow("tyre", imgcir)
cv.waitKey(0)  
# to draw a lot of circles 
for i in range(5):
    (xc,yc),radius = cv.minEnclosingCircle(contours[i])
    center = (int(xc),int(yc))
    radius = int(radius)
    imgcircle = cv.circle(crop_img,center,radius,(0,255,0),2)
cv.imshow("tyre", imgcircle)
cv.waitKey(0)    

#histogram 
from matplotlib import pyplot as plt
kingimg = cv.imread('Kingfisher.png')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([kingimg],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
# Harris Corner Detector
colimage=cv.imread('tallestbuilding.png')
graytall = cv.cvtColor(colimage,cv.COLOR_BGR2GRAY)
dst = cv.cornerHarris(graytall,2,3,0.04)
dst = cv.dilate(dst,None)
max=0.001*dst.max()
colimage[dst>max]=[0,0,255]
cv.imshow('dst',colimage)
cv.waitKey(0)

#Brute-Force matcher 
img1 = cv.imread('tyre.jpg',0)          # queryImage
img2 = cv.imread('crop_img.png',0) # trainImage

# Initiate SIFT detector
orb = cv.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 5 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:25],None,flags=2)
plt.imshow(img3),plt.show()



