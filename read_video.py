import cv2 as cv
# you can reference to camera, 0, 1 , 2
capture = cv.VideoCapture('Video/Echo/echo1.mp4')

def changeRes(width, height):
    #only Live Video
    #property of capture
    capture.set(3, width)
    capture.set(4, height)
    
      
def rescaleFrame(frame, scale=0.75):
    #Image, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

while True:
    #read frame by frame, and bool if successfully read
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame)
    cv.imshow('Video', frame)
    cv.imshow('Video_resize', frame_resized)
    #if letter d press than break up loop
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()