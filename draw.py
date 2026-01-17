import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow('Blank', blank)

# 1. Paint the image a certain colour

# blank[:] = 0,255,0
# cv.imshow('Green', blank)

# 1. Paint a certain part of the image
blank[200:300, 300:400] = 0,0, 255
cv.imshow('Green', blank)

# 2. Draw a Rectangle
#                   origin pt, accross to what pt, color, thickness
#cv.rectangle(blank, (0,0), (250, 250), (0, 255, 0), thickness=2)
# to fill is thickness=cv.FILLED or thickness=-1
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness=2)
cv.imshow('Rectangle', blank)

# img = cv.imread('Image/cat.jpg')
# cv.imshow('Cat', img)

cv.waitKey(0)