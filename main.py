import math
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
from cv2 import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

cap = cv2.VideoCapture(0)
image = cv2.imread("orange_5.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thres = 30
ret,bw = cv2.threshold(gray,thres,255,cv2.THRESH_BINARY)
gray = cv2.GaussianBlur(bw, (7, 7), 0)
cv2.imshow("test",gray)

edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

for c in cnts:
    if cv2.contourArea(c) < 100:
        continue

    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 64), 2)


    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 255, 64), -1)

    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    cv2.circle(orig, (int(tltrX), int(tltrY)), 0, (0, 255, 64), 0)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 0, (0, 255, 64), 0)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 0, (0, 255, 64), 0)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 0, (0, 255, 64), 0)

    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 255, 255), 1)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 255, 255), 1)

    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    if pixelsPerMetric is None:
        pixelsPerMetric = dB / 1.02362205

    dimA = (dA / pixelsPerMetric) * 2.54
    dimAcen = math.floor(dimA)
    print(dimAcen)
    if dimAcen >= 7:
        cv2.putText(orig, "{:.0f}cm Size:00".format(dimAcen),
        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)
    elif dimAcen == 6:
        cv2.putText(orig, "{:.0f}cm Size:1".format(dimAcen),
        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)
    elif dimAcen == 5:
        cv2.putText(orig, "{:.0f}cm Size:3".format(dimAcen),
        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)
    else:
        cv2.putText(orig, "{:.0f}cm Size:Small Size!!".format(dimAcen),
        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)
    cv2.imshow("Measuring_Size_Image", orig)
    cv2.waitKey(0)