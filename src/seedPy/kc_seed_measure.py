import cv2
import numpy as np
import skimage
import os
import shapely
from shapely.geometry import Point, Polygon, box
from itertools import compress
import random as rng
import pandas as pd


def showImage(img):

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

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

def setRoi(points, mask_shape):
    """
    Takes an array of points and returns a mask to use to use as an ROI
    """

    import cv2
    import numpy as np

    x, y, w, h  = cv2.boundingRect(points)
    blank = np.zeros(mask_shape, dtype = 'uint8')
    roi_mask = cv2.rectangle(blank, (x, y), (x+w, y+h), 255, -1)
    canny = cv2.Canny(roi_mask, 100, 200)
    roi_cont, _ = cv2.findContours(canny,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return roi_mask, roi_cont

def getMaskedContours(src, roi):
    """
    takes a color src image and a binary mask
    returns a
    """
    assert type(src)== np.ndarray
    assert src.shape[0:2] == roi.shape

    kernel = np.ones((5, 5), np.uint8)

    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 0, 120])
    upper = np.array([179, 255, 255])
    bin_img = cv2.inRange(hsv_img, lower, upper)
    showImage(bin_img)
    inverted = np.invert(bin_img)
    masked = cv2.bitwise_and(src, src, mask = roi)
    showImage(masked)
    canny = cv2.Canny(masked, 100, 200)
    showImage(canny)
    closed_edge = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    showImage(closed_edge)
    #noise_removal = cv2.morphologyEx(closed_edge, cv2.MORPH_OPEN, kernel)

    contours, __ = cv2.findContours(closed_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

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
    return meas_series

def getSeedColor(src, seed_cont):
    """

    """
    conts = tuple(seed_cont)
    mask = np.zeros(src.shape[0:3], np.uint8)
    cv2.drawContours(src, conts, -1, (0, 255, 0), -1)

    showImage(src)

    return cv2.findNonZero(mask)


if os.path.exists('measurement_files'):
    pass
else:
    os.mkdir('measurement_files')

if os.path.exists('output_images'):
    pass
else:
    os.mkdir('output_images')

for root, dirs, files in os.walk('images/'):
    for file in files:
        raw = cv2.imread(root + '/' + file)
        ref_points = detectReferenceCircles(raw)
        contours = closedContours(raw, 100, 200, 5)
        cont_idx = contoursTest(ref_points, contours)
        filtered_conts = tuple(compress(contours, cont_idx)) # remove all of the contours outside of the ROI
        meas_df = pd.DataFrame(columns=['convex_hull_area', 'perimeter', 'length', 'width', 'extent', 'solidity', 'equi_diameter', 'aspect_ratio'], dtype = np.float64)

        for i, fc in enumerate(filtered_conts):
            if fc.shape[0] > 20: # filter out small, non-seed objects
                color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                measurements = measureSeed(fc)
                meas_df = meas_df.append(measurements, ignore_index=True)
                minEllipse = cv2.fitEllipse(fc)
                cv2.ellipse(raw, minEllipse, color, 2)
                min_rect = cv2.minAreaRect(fc)
                bbox = cv2.boxPoints(min_rect)
                bbox = np.intp(bbox) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
                cv2.drawContours(raw, [bbox], 0, color)

        meas_df.to_csv(f"measurement_files/{file}_measurements.csv")
        cv2.imwrite(f"output_images/{file}_processed.JPG", raw)


        print(f'{file} complete!')

