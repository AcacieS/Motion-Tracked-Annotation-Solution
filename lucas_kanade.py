import cv2 as cv
import numpy as np

ix, iy = 200, 200
refresh = False
point_selected = False

def onMouse(event, x, y, flags, param):
    global ix, iy, refresh, point_selected
    if event == cv.EVENT_LBUTTONDOWN:
        refresh = True
        ix, iy = x, y
        point_selected = True
        print("Point selected:", ix, iy)

cv.namedWindow("window")
cv.setMouseCallback("window", onMouse)
cap = cv.VideoCapture('Video/Lapchole/Lapchole2.mp4')

old_gray = None
old_pts = None
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # If point selected, track it
    if point_selected:
        if  refresh:
            old_gray = gray.copy()
            old_pts = np.array([[ix, iy]], dtype=np.float32).reshape(-1,1,2)
            refresh = False
        else:
            new_pts, status, err = cv.calcOpticalFlowPyrLK(
                old_gray, gray, old_pts, None,
                maxLevel=1,
                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 15, 0.08)
            )

            x, y = new_pts.ravel()
            cv.circle(frame, (int(x), int(y)), 5, (0,255,0), -1)

            old_gray = gray.copy()
            old_pts = new_pts

    cv.imshow("window", frame)

    if cv.waitKey(1) == 27:  # ESC
        break

cap.release()
cv.destroyAllWindows()