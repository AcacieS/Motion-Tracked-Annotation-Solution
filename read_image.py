import cv2 as cv

img = cv.imread('Image/cat_large.jpg')



def rescaleFrame(frame, scale=0.2):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img_resized = rescaleFrame(img)
#show image in window: cv.imshow(name of the window, matrix of pixels to display)
cv.imshow('Cat', img_resized)

#wait for delay for key to be presss
cv.waitKey(0)