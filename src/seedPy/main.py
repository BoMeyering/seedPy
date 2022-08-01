from process import *
import os
import subprocess
import random as rng

if not os.path.exists('log_files'):
	os.mkdir('log_files')
	subprocess.call(['touch', 'log_files/.log'])

if not os.path.exists('measurement_files'):
	os.mkdir('measurement_files')

with open('log_files/.log', 'w') as f:
	for root, dirs, files in os.walk('images'):
		for file in files:
			src = cv2.imread(f'images/{file}')
			ref_points = detectReferenceCircles(src)
			# print(ref_points)
			ref_cp = src.copy()
			try:
				assert ref_points.shape == (4, 1, 2)
			except AssertionError:
				print(f"Only detected {ref_points.shape[0]} reference circles in {file}.JPG")
				f.writelines(f"{file}: Error - only detected {ref_points.shape[0]} reference circles\n")
				if ref_points.shape[0] > 2:
					for pt in ref_points[:,0,:]:
						cv2.circle(ref_cp, (pt[0], pt[1]), 20, (255, 0, 0), -1)
					cv2.imwrite(f"detected_circles/{file}_reference.JPG", ref_cp)
					print(f"Wrote {file}_reference.JPG with {ref_points.shape[0]}. Continuing with analysis")
				else:
					continue
			for pt in ref_points[:,0,:]:
				cv2.circle(ref_cp, (pt[0], pt[1]), 20, (255, 0, 0), -1)

			cv2.imwrite(f"detected_circles/{file}_reference.JPG", ref_cp)
			print(f"Wrote {file}_reference.JPG")

			# Identify all contours in the src image, and save only the contours in the ROI
			src_contours = closedContours(src, 100, 200, 5)
			cnt_idx = contoursTest(ref_points, src_contours)
			roi_cnt = tuple(compress(src_contours, cnt_idx))

			try:
				crop_cnt = np.concatenate(roi_cnt)
			except ValueError:
				f.writelines(f"{file}: Error - image does not contain any recognizable contours within the ROI\n")
				meas_df = pd.DataFrame(columns=['convex_hull_area', 'perimeter', 'length', 'width', 'extent', 'solidity', 'equi_diameter', 'aspect_ratio'], dtype = np.float64)
				meas_df.to_csv(f"measurement_files/{file}_measurements.csv")
				continue


			cropped_src, crop_coords = cropSrc(src, crop_cnt)

			crop_cp = cropped_src.copy()

			# showImage(crop_cp)
			# cropped_edges = cv2.Canny(crop_cp, 100, 200)
			# cropped_conts, _ = cv2.findContours(cropped_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			
			cropped_conts = closedContours(crop_cp, 100, 200, 5)

			cropped_mask = np.zeros(crop_cp.shape[0:2], dtype = np.uint8)
			mask_cp = cropped_mask.copy()

			cv2.drawContours(mask_cp, cropped_conts, -1, 255, 2)
			cv2.drawContours(cropped_mask, cropped_conts, -1, 255, cv2.FILLED)

			cv2.imwrite(f"contour_output/{file}_thin.JPG", mask_cp)
			cv2.imwrite(f"contour_output/{file}_filled.JPG", cropped_mask)

			dist = ndimage.distance_transform_edt(cropped_mask)

			# localMax = peak_local_max(dist, indices = False, min_distance=10)

			# markers = ndimage.label(localMax, structure = np.ones((3, 3)))[0]

			peak_coord = peak_local_max(dist, min_distance=10, labels = cropped_mask)
			peak_mask = np.zeros_like(dist, dtype=bool)
			peak_mask[tuple(peak_coord.T)] = True
			markers = ndimage.label(peak_mask, structure = np.ones((3, 3)))[0]

			labels = watershed(-dist, markers, mask = cropped_mask)

			# print(labels)
			# print(len(np.unique(labels)))
			# mask = np.zeros(crop_cp.shape, dtype = np.uint8)
			# for label in np.unique(labels):
			# 	if label == 0:
			# 		continue
			# 	color = (rng.randint(0, 255),rng.randint(0, 255),rng.randint(0, 255))
			# 	mask[labels == label] = color
			# showImage(mask)

			mask = np.zeros(crop_cp.shape[0:2], dtype = np.uint8)
			meas_df = pd.DataFrame()
			for label in np.unique(labels):
				if label == 0:
					continue
				color = (rng.randint(0, 255),rng.randint(0, 255),rng.randint(0, 255))
				mask = np.zeros(crop_cp.shape[0:2], dtype = np.uint8)
				mask[labels == label] = 255
				cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				if cv2.contourArea(cnts[0]) < 50:
					continue
				measurements = measureSeed(cnts[0])
				meas_df = pd.concat([meas_df, measurements]).reset_index(drop=True)
				cv2.drawContours(crop_cp, cnts, -1, (255, 0, 0), 1)
			cv2.imwrite(f'output/{file}_processed.JPG', crop_cp)
			meas_df.to_csv(f"measurement_files/{file}_measurements.csv")



















# for root, dirs, files in os.walk('images'):
# 	for file in files:
# 		src = cv2.imread(f'images/{file}')
# 		ref_points = detectReferenceCircles(src)
# 		# print(ref_points)
# 		ref_cp = src.copy()
# 		try:
# 			assert ref_points.shape == (4, 1, 2)
# 		except AssertionError:
# 			print(f"Only detected {ref_points.shape[0]} reference circles in {file}.JPG.\nSkipping for now")
# 			with open('log_files/.log', 'w') as f:
# 				f.writelines(f"{file}: Error - only detected {ref_points.shape[0]} reference circles")
# 			continue
# 		for pt in ref_points[:,0,:]:
# 			cv2.circle(ref_cp, (pt[0], pt[1]), 20, (255, 0, 0), -1)

# 		cv2.imwrite(f"detected_circles/{file}_reference.JPG", ref_cp)
# 		print(f"Wrote {file}_reference.JPG")

# 		# Identify all contours in the src image, and save only the contours in the ROI
# 		src_contours = closedContours(src, 100, 200, 5)
# 		cnt_idx = contoursTest(ref_points, src_contours)
# 		roi_cnt = tuple(compress(src_contours, cnt_idx))

# 		try:
# 			crop_cnt = np.concatenate(roi_cnt)
# 		except ValueError:
# 			with open('log_files/.log', 'w') as f:
# 				f.writelines(f"{file}: Error - image does not contain any recognizable contours within the ROI")
# 			meas_df = pd.DataFrame(columns=['convex_hull_area', 'perimeter', 'length', 'width', 'extent', 'solidity', 'equi_diameter', 'aspect_ratio'], dtype = np.float64)
# 			meas_df.to_csv(f"measurement_files/{file}_measurements.csv")
# 			continue


# 		cropped_src, crop_coords = cropSrc(src, crop_cnt)

# 		crop_cp = cropped_src.copy()

# 		# showImage(crop_cp)
# 		cropped_edges = cv2.Canny(crop_cp, 100, 200)
# 		cropped_conts, _ = cv2.findContours(cropped_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 		cropped_mask = np.zeros(crop_cp.shape[0:2], dtype = np.uint8)
# 		cv2.drawContours(cropped_mask, cropped_conts, -1, 255, cv2.FILLED)
# 		# showImage(cropped_mask)
# 		dist = ndimage.distance_transform_edt(cropped_mask)
# 		# print(dist)
# 		# showImage(dist)

# 		localMax = peak_local_max(dist, indices = False, min_distance=10)

# 		markers = ndimage.label(localMax, structure = np.ones((3, 3)))[0]

# 		labels = watershed(-dist, markers, mask = cropped_mask)

# 		# print(labels)
# 		# print(len(np.unique(labels)))
# 		# mask = np.zeros(crop_cp.shape, dtype = np.uint8)
# 		# for label in np.unique(labels):
# 		# 	if label == 0:
# 		# 		continue
# 		# 	color = (rng.randint(0, 255),rng.randint(0, 255),rng.randint(0, 255))
# 		# 	mask[labels == label] = color
# 		# showImage(mask)

# 		mask = np.zeros(crop_cp.shape[0:2], dtype = np.uint8)
# 		# meas_df = pd.DataFrame(columns=['convex_hull_area', 'perimeter', 'length', 'width', 'extent', 'solidity', 'equi_diameter', 'aspect_ratio'], dtype = np.float64)
# 		meas_df = pd.DataFrame()
# 		for label in np.unique(labels):
# 			if label == 0:
# 				continue
# 			color = (rng.randint(0, 255),rng.randint(0, 255),rng.randint(0, 255))
# 			mask = np.zeros(crop_cp.shape[0:2], dtype = np.uint8)
# 			mask[labels == label] = 255
# 			cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 			if cv2.contourArea(cnts[0]) < 40:
# 				continue
# 			measurements = measureSeed(cnts[0])
# 			# meas_df = meas_df.append(measurements, ignore_index=True)
# 			meas_df = pd.concat([meas_df, measurements]).reset_index(drop=True)
# 			cv2.drawContours(crop_cp, cnts, -1, color, 1)
# 		cv2.imwrite(f'output/{file}_processed.JPG', crop_cp)
# 		meas_df.to_csv(f"measurement_files/{file}_measurements.csv")


# 		# dist_cp = cv2.normalize(dist, dist, 50, 100, cv2.NORM_MINMAX)
# 		# showImage(dist_cp)

# 		# hsv_img = cv2.cvtColor(cropped_src, cv2.COLOR_BGR2HSV)
# 		# lower = np.array([50, 0, 120])
# 		# upper = np.array([179, 255, 255])
# 		# bin_img = cv2.inRange(hsv_img, lower, upper)
# 		# inverted = cv2.bitwise_not(bin_img)

# 		# labels = getLabels(inverted)
# 		# labeled = drawLabels(labels, crop_cp)

# 		# cv2.imwrite(f"output2/{file}_labeled.JPG", labeled)
