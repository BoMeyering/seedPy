import cv2
import sklearn
import skimage
import os
import numpy as np
from scipy import ndimage
import sys
import shapely
from shapely.geometry import Point, Polygon, box
from itertools import compress
import random as rng
import pandas as pd
from skimage.feature import peak_local_max

np.set_printoptions(threshold=sys.maxsize)


def detectReferenceCircles(src):
    """
    Function to take a raw image as a numpy array
    Detects large circular blobs
    Returns a numpy array of center points for each reference circle
    """

    import cv2
    import numpy as np

    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)        
    lower = np.array([0, 0, 120])
    upper = np.array([179, 255, 255])
    bin_img = cv2.inRange(hsv_img, lower, upper)

    params = cv2.SimpleBlobDetector_Params() # initialize parameters
    params.filterByArea = True
    params.maxArea = 500000
    params.minArea = 10000
    params.filterByCircularity = True
    params.minCircularity = .8 # filter for circles

    detector = cv2.SimpleBlobDetector_create(params) # intialize the detector
    keypoints = detector.detect(bin_img)

    points = np.array(np.int32([keypoint.pt for keypoint in keypoints]))
    dim1 = points.shape[0]
    points = points.ravel().reshape((dim1, 1, 2))

    return points


def contoursTest(roi_points, contours):
    """
    takes a numpy array of points for the roi
    computes a shapely Polygon object for the roi
    iteratively tests each contour in contours for containment in roi_poly
    returns a 1D numpy array of boolean values for contours that are in the roi
    """

    x, y, w, h = cv2.boundingRect(roi_points) # get ROI bounding box
    roi_poly = box(y, x, y+h, x+w)

    results = np.zeros((len(contours)), dtype = int)
    for idx, c in enumerate(contours):
        if cv2.contourArea(c) < 20:
            continue
        pt_list = [(x, y) for x, y in zip(c[:, :, 1].ravel(), c[:, :, 0].ravel())]
        if len(pt_list) < 3:
            continue
        cont_poly  = Polygon(pt_list)
        results[idx] = roi_poly.contains(cont_poly)

    return results.tolist()


def closedContours(src, threshold1, threshold2, kernel_size):
    """
    Takes a src image
    Detects edges and closes any gaps
    Finds and returns all external contours on the closed edges
    """

    assert type(src)==np.ndarray

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    edges = cv2.Canny(src, threshold1, threshold2)
    closed_edge = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def showImage(src):
	cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
	cv2.imshow('image', src)
	cv2.waitKey(0)
	cv2.destroyWindow('image')



# for root, dirs, files in os.walk('images/'):
	# print(files)
stuff = os.walk('images/')

src = cv2.imread('images/2650.JPG')

# showImage(src)


ref_points = detectReferenceCircles(src)
contours = closedContours(src, 100, 200, 5)
cont_idx = contoursTest(ref_points, contours)
filtered_conts = tuple(compress(contours, cont_idx))

print(ref_points)
x, y, w, h = cv2.boundingRect(ref_points)
# print(bbrect)

src_cp = src.copy()

cv2.rectangle(src_cp, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.drawContours(src_cp, filtered_conts, -1, (0, 0, 255), 1)

showImage(src_cp)


# print(len(filtered_conts))

cnts = np.concatenate(filtered_conts)
# print(cnts.shape)

x, y, w, h = cv2.boundingRect(cnts)
crop_dims = [(x-10, y-10), (x+w+10, y+h+10)]

cropped = src[crop_dims[0][1]:crop_dims[1][1], crop_dims[0][0]:crop_dims[1][0]]

showImage(cropped)
# showImage(src)

# showImage(src)






shifted = cv2.pyrMeanShiftFiltering(cropped, 10, 30)

showImage(shifted)

hsv = cv2.cvtColor(shifted, cv2.COLOR_BGR2HSV)
# gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

# showImage(gray)

# # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_OTSU)[1]
# showImage(thresh)

hsv_thresh = cv2.inRange(hsv, (0, 0, 0), (40, 255, 255))

# showImage(hsv_thresh)

kernel = np.ones((5, 5), np.uint8)

opening = cv2.morphologyEx(hsv_thresh, cv2.MORPH_OPEN, kernel)

# showImage(opening)

sure_bg = cv2.dilate(opening, kernel, iterations = 3)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
# print(dist_transform)
print(dist_transform.max())
ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# showImage(dist_transform)
# showImage(sure_bg)
# showImage(sure_fg)
# showImage(unknown)
# showImage(ret)




# D = ndimage.distance_transform_edt(opening)

# print(D.shape, dist_transform.shape)
# localMax = peak_local_max(D, indices=False, min_distance=5,
#     labels=thresh)

# print(localMax)

# showImage(D)

# dist = cv2.distanceTransform(opening,cv2.DIST_L2,3)

# showImage(dist)

# showImage(dist)
# cv2.normalize(dist, dist, 0, 100, cv2.NORM_MINMAX)

# showImage(dist)

# _, dist = cv2.threshold(dist, 0, 5, cv2.THRESH_BINARY)

# showImage(dist)

# print(sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)

markers = markers+1


# print(markers)
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

# print(markers)

markers = cv2.watershed(cropped,markers)

# print(markers)

for obj in np.unique(markers)[0:30]:
    print(obj)
    if obj <= 1:
        continue
    mask = np.zeros(cropped.shape, dtype="uint8")
    # showImage(mask)
    mask[markers == obj] = 255
    showImage(mask)

    # cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # cnts = imutils.grab_contours(cnts)
    # # c = max(cnts, key=cv2.contourArea)
    # black = np.zeros(cropped.shape, dtype = np.uint8)
    # cv2.drawContours(black, cnts, -1, (255), 2)
    # showImage(black)


# print(markers.shape)
# print(src.shape)

cropped[markers == -1] = [255,0,0]

showImage(cropped)
