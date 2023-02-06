# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:09:44 2018

@author: Admin
"""

import cv2 as cv
import numpy as np
image=cv.imread("football.png")
#cv.imshow("image", image)
#cv.waitKey(0)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#cv.imshow("image", gray)
#cv.waitKey(0)
gray = cv.GaussianBlur(gray,(5,5),2)
gray = cv.medianBlur(gray,5)
gray = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,3.5)
kernel = np.ones((3,3),np.uint8)
gray = cv.dilate(gray,kernel,iterations = 1)
circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1.9, 200, param1=30, param2=65, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
image1=image
for (x, y, r) in circles:
    # draw the outer circle
   if r<=45:    
       cv.circle(image1, (x, y), r, (0, 255, 0), 4)
   
cv.imshow('detected circles',image1)
cv.waitKey(0)
cv.destroyAllWindows()


