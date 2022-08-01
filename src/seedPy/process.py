import cv2
import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, box
from itertools import compress
import random as rng
import sys
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import imutils

# np.set_printoptions(threshold=sys.maxsize)


def showImage(src):
	cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
	cv2.imshow('image', src)
	cv2.waitKey(0)
	cv2.destroyWindow('image')


def detectReferenceCircles(src):
    """
    Function to take a raw image as a numpy array
    Detects large circular blobs
    Returns a numpy array of center points for each reference circle
    """

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
        if cv2.contourArea(c) < 50:
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


def cropSrc(src, points, buffer = 10):
	"""
	Takes a source image and a set of points belonging to a contour or a concatenated set of contours
	Finds the smallest bounding box that contains the points
	Crops the src image according to the bounding box with a buffer zone in pixels
	Returns cropped image array and 
	"""

	x, y, w, h = cv2.boundingRect(points)
	crop_coords = np.array([[x-buffer, y-buffer], [x+w+buffer, y+h+buffer]])

	cropped = src[crop_coords[0][1]:crop_coords[1][1], crop_coords[0][0]:crop_coords[1][0]]

	return (cropped, crop_coords)


def labelObjects(src_bin, kernel_size):
	"""
	Takes a binary source image with black background and white foreground
	Finds the object markers, performes a connected component analysis
	Uses watershed to return an array of the labeled objects
	"""
	assert kernel_size in [3, 5, 7]

	kernel = np.ones((kernel_size, kernel_size), np.uint8)

	dist = cv2.distanceTransform(src_bin, cv2.DIST_L2, 3)

	# localMax = peak_local_max(dist, indices = False, min_distance=20, labels = src_bin)
	# print(localMax)

	# markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
	# labels = watershed(-dist, markers, mask = src_bin)
	# print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

	background = cv2.dilate(src_bin, kernel, iterations=3)
	_, foreground = cv2.threshold(dist, 0.4*dist.max(), 255, 0)
	foreground = np.uint8(foreground)

	unk_zone = cv2.subtract(background, foreground)

	showImage(background)
	showImage(foreground)
	
	_, markers = cv2.connectedComponents(foreground)



	markers+=1

	markers[unk_zone==255]=0
	print(markers)

	markers = cv2.watershed(src_bin, markers)

	return labels


def getLabels(src_bin, min_distance = 10):
	"""
	Takes a binary source image (foreground = 255, background = 0)
	Calculates the distance transform matrix to from nonzero pixels to nearest 0
	Finds local maxima of peaks separated by min_distance
	Returns a numpy array of labels of shape src_bin
	"""
	dist = ndimage.distance_transform_edt(src_bin)

	# localMax = peak_local_max(dist, indices = False, min_distance=min_distance, labels = src_bin)
	# markers = ndimage.label(localMax, structure = np.ones((3, 3)))[0]

	peak_coord = peak_local_max(dist, min_distance=min_distance, labels = src_bin)
	peak_mask = np.zeros_like(dist, dtype=bool)
	peak_mask[tuple(peak_coord.T)] = True
	markers = ndimage.label(peak_mask, structure = np.ones((3, 3)))[0]

	labels = watershed(-dist, markers, mask = src_bin)

	return labels


def drawLabels(labels, src):
	"""
	Takes a numpy array of  unique object labels (0 = background, integer = object)
	Filters the individual contours of each object based on an area criteria
	Draws the contours on the object as well as the object number
	Returns an image array
	"""

	for label in np.unique(labels):
		if label == 0:
			continue
		mask = np.zeros(labels.shape, dtype= np.uint8)
		mask[labels == label] = 255
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# print(cnts)
		cnts = imutils.grab_contours(cnts)
		# print(cnts)
		c = max(cnts, key=cv2.contourArea)
		if cv2.contourArea(c) <= 50:
			continue
		((x, y), r) = cv2.minEnclosingCircle(c)
		# cv2.drawContours(src, c, -1, (255,0,0), 2)
		# cv2.circle(src, (int(x), int(y)), int(r), (0, 255, 0), 2)
		cv2.putText(src, "{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
	
	return src

def measureSeed(seed_cont):
    """
    Area, perimeter, min_rect (LxW), Solidity, Convexity Deflects, Avg color, Aspect Ratio, Extent
    Eq.Diameter, 
    """
    area = cv2.contourArea(seed_cont)
    perimeter = cv2.arcLength(seed_cont, closed=True)
    (x, y), (W, L), orAngle = cv2.fitEllipse(seed_cont)
    _, _, w, h = cv2.boundingRect(seed_cont)
    extent = float(area)/(w*h) # ratio of contour area to bounding box area
    hull = cv2.convexHull(seed_cont)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area # ratio of contour area to convex hull area
    equi_diameter = np.sqrt(4*area/np.pi)
    aspect_ratio = W/L

    meas_series = pd.Series({
        "area": area,
        "convex_hull_area": hull_area,
        "perimeter": perimeter,
        "length": L,
        "width": W, 
        "extent": extent,
        "solidity": solidity,
        "equi_diameter": equi_diameter,
        "aspect_ratio": aspect_ratio
        })

    meas_df = pd.DataFrame([[area, hull_area, perimeter, L, W, extent, solidity, equi_diameter, aspect_ratio]],
    	columns = ['area', 'convex_hull_area', 'perimeter', 'length', 'width', 'extent', 'solidity', 'equi_diameter', 'aspect_ratio'])

    # return meas_series
    return meas_df