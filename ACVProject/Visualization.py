"""
Visualization
Author: Tim Burt
This library holds plot/movie generation functions for the package and the automated lung masking feature.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from skimage import morphology
from sklearn.cluster import KMeans
import csv


def update_hu_range(img, cur_min, cur_max):
	"""Reads absolute min/max of image pixels and compares them to current min/max values as input"""
	local_min_hu = np.amin(img)
	local_max_hu = np.amax(img)
	if local_min_hu < cur_min:
		cur_min = local_min_hu
	if local_max_hu > cur_max:
		cur_max = local_max_hu
	return cur_min, cur_max


def apply_lungmask(img, mask):
	"""Applies erode/dilate mask to image to remove lungs"""
	img_shape = img.shape  # should be 256x256
	img_masked = np.ma.where(mask == 1.0, img, np.amin(img))  # sets region outside mask to same minimum as outside crop
	return img_masked


def make_lungmask(img):
	"""Creates erode/dilate masks for an input image to remove lungs"""
	row_size = img.shape[0]
	col_size = img.shape[1]
	mean = np.mean(img)
	std = np.std(img)
	img = img - mean
	img = img / std
	# Find the average pixel value near the lungs
	# to renormalize washed out images
	middle = img[int(col_size / 5):int(col_size / 5 * 4),
	         int(row_size / 5):int(row_size / 5 * 4)]  # FIXME: doesn't work for projection
	mean = np.mean(middle)
	# To improve threshold finding, I'm moving the
	# underflow and overflow on the pixel spectrum
	img[img == max] = mean
	img[img == min] = mean
	#
	# Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
	#
	kmeans = KMeans(n_clusters=10).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
	centers = sorted(kmeans.cluster_centers_.flatten())
	threshold = np.mean(centers)

	thresh_img = np.where(img > threshold, 1.0, 0.0)  # sets area outside heart to 0, inside to 1
	eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
	dilation = morphology.dilation(eroded, np.ones([6, 6]))
	return dilation


def make_axial_movie(image, cmap, movie_fn="axial_movie_projection", fps=8):
	num_slices = image.shape[0]  # assuming square
	slices = np.squeeze(image)
	# find global voxel max/min and set colorbar to that fixed range
	max_all = np.max(np.max(slices))
	min_all = np.min(np.min(slices))
	for i in range(num_slices):
		if (i % int(num_slices / 10)) == 0:
			print("Plotting slice %d of %d..." % (i + 1, num_slices))
		slice = slices[:][:][i]
		plt.imshow(slice, cmap=plt.get_cmap(cmap), vmin=min_all, vmax=max_all)
		plt.colorbar()
		plt.title("Affine Method\nAxial slice %d of %d\nHounsfield Units" % (i + 1, num_slices))
		plt.xlabel("X (pixels)")
		plt.ylabel("Y (pixels)")
		plt.savefig("slice_%d.png" % i)
		plt.close()

	print("Creating movie...")
	os.system("ffmpeg -r %d -i slice_%%d.png -r 30 %s.mp4" % (fps, movie_fn))


def make_axial_movie_comparison(affine_image, projection_image, masked_affine_image, masked_projection_image,
                                cmap, movie_fn, patient_id, fps=8, lung_mask=False):
	if lung_mask:
		num_plots = [2, 2]
		fig_size = [18, 18]
	else:
		num_plots = [1, 2]
		fig_size = [18, 9]

	num_slices = affine_image.shape[0]  # assuming square and both images same size

	affine_slices = np.squeeze(affine_image)
	masked_affine_slices = np.squeeze(masked_affine_image)

	# find affine_image global voxel max/min and set colorbar to that fixed range
	affine_max_all = np.max(np.max(affine_slices))
	affine_min_all = np.min(np.min(affine_slices))

	projection_slices = np.squeeze(projection_image)
	masked_projection_slices = np.squeeze(masked_projection_image)

	# find projection_image global voxel max/min and set colorbar to that fixed range
	projection_max_all = np.max(np.max(projection_slices))
	projection_min_all = np.min(np.min(projection_slices))

	for i in range(num_slices):
		if (i % int(num_slices / 10)) == 0:
			print("Plotting slice %d of %d..." % (i + 1, num_slices))

		affine_slice = affine_slices[:][:][i]
		projection_slice = projection_slices[:][:][i]

		masked_affine_slice = masked_affine_slices[:][:][i]
		masked_projection_slice = masked_projection_slices[:][:][i]

		plt.figure(figsize=fig_size)
		# affine plot on left
		plt.subplot(num_plots[0], num_plots[1], 1)
		plt.imshow(affine_slice, cmap=plt.get_cmap(cmap), vmin=affine_min_all, vmax=affine_max_all)
		cbar1 = plt.colorbar()
		cbar1.ax.set_ylabel("Hounsfield Units (HU)", rotation=90)
		plt.title("Affine Method\nAxial slice %d of %d\nPatient ID %s" % (i + 1, num_slices, patient_id))
		plt.xlabel("X (pixels)")
		plt.ylabel("Y (pixels)")

		# projection plot on right
		plt.subplot(num_plots[0], num_plots[1], 2)
		plt.imshow(projection_slice, cmap=plt.get_cmap(cmap), vmin=projection_min_all, vmax=projection_max_all)
		cbar2 = plt.colorbar()
		cbar2.ax.set_ylabel("Hounsfield Units (HU)", rotation=90)
		plt.title("Projection Method\nAxial slice %d of %d\nPatient ID %s" % (i + 1, num_slices, patient_id))
		plt.xlabel("X (pixels)")
		plt.ylabel("Y (pixels)")

		if lung_mask:
			plt.subplot(num_plots[0], num_plots[1], 3)
			# affine_mask_slice = make_lungmask(affine_slice)
			plt.imshow(masked_affine_slice, cmap=plt.get_cmap(cmap), vmin=affine_min_all, vmax=affine_max_all)
			cbar3 = plt.colorbar()
			cbar3.ax.set_ylabel("Hounsfield Units (HU)", rotation=90)
			plt.title("Affine After Erosion & Dilation\nAxial slice %d of %d\nPatient ID %s" % (
				i + 1, num_slices, patient_id))
			plt.xlabel("X (pixels)")
			plt.ylabel("Y (pixels)")

			plt.subplot(num_plots[0], num_plots[1], 4)
			# projection_mask_slice = make_lungmask(projection_slice)
			plt.imshow(masked_projection_slice, cmap=plt.get_cmap(cmap), vmin=projection_min_all,
			           vmax=projection_max_all)
			cbar4 = plt.colorbar()
			cbar4.ax.set_ylabel("Hounsfield Units (HU)", rotation=90)
			plt.title("Projection After Erosion & Dilation\nAxial slice %d of %d\nPatient ID %s" % (
				i + 1, num_slices, patient_id))
			plt.xlabel("X (pixels)")
			plt.ylabel("Y (pixels)")

		plt.savefig("slice_%d.png" % i)
		plt.close()

	print("Creating movie...")
	os.system("ffmpeg -r %d -i slice_%%d.png -r 30 %s.mp4 -y" % (fps, movie_fn))


def plot_tf_accuracy(data_path):
	""".csv files exported using TensorBoard"""

	os.chdir(data_path)
	test_accuracy_fns = sorted(glob.glob("*test_accuracy.csv"))
	train_accuracy_fns = sorted(glob.glob("*train_accuracy.csv"))

	### THESE WILL CHANGE DEPENDING ON YOUR RESULTS FILES, PLEASE CHECK ###
	plot_labels = ["ResNet18, affine input, W lungmask, 16CH axial averaging",
	               "ResNet18, projection input, W/O lungmask, 4CH axial averaging",
	               "ResNet18, projection input, W/O lungmask, 16CH axial averaging",
	               "ResNet34, affine input, W lungmask, 8CH axial averaging",
	               "ResNet34, projection input, W/O lungmask, 8CH axial averaging",
	               "ResNet101, affine input, W/O lungmask, 4CH axial averaging",
	               "ResNet101, affine input, W lungmask, 4CH axial averaging"]

	plot_linecolors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	plot_linemarkers = ['^', 'o', 'o', '^', 'o', 'o', '^']
	plot_linewidths = [1, 3, 1, 2, 2, 3, 3]
	########################################################################

	test_accuracy_data = []
	train_accuracy_data = []

	# assuming number of test/train accuracy files are the same
	for i in range(len(test_accuracy_fns)):
		with open(test_accuracy_fns[i], "r") as f:
			reader = csv.reader(f)
			test_accuracy_data.append(list(reader)[1:])
		f.close()

		with open(train_accuracy_fns[i], "r") as f:
			reader = csv.reader(f)
			train_accuracy_data.append(list(reader)[1:])
		f.close()

	plt.figure(figsize=(28, 12))

	for i in range(len(train_accuracy_data)):
		print("Plotting training/testing results %d of %d..." % (i + 1, len(train_accuracy_data)))

		train_epoch_temp = np.asarray(train_accuracy_data[i], dtype=float)[:, 1]
		train_accuracy_temp = np.asarray(train_accuracy_data[i], dtype=float)[:, 2]

		plt.subplot(1, 2, 1)
		plt.plot(train_epoch_temp, train_accuracy_temp, label=plot_labels[i], color=plot_linecolors[i],
		         marker=plot_linemarkers[i], lw=plot_linewidths[i] * 2, markersize=plot_linewidths[i] * 3)
		plt.xlabel("Standard epochs")
		plt.ylabel("Training accuracy (%)")
		plt.title(
			"Med3DResNet training accuracy vs. epochs\n70/30 train/test (102/28), batch_size=40, initial_learning rate=0.1, 25 epochs")
		plt.legend()
		plt.ylim(0.0, 1.0)

		test_epoch_temp = np.asarray(test_accuracy_data[i], dtype=float)[:, 1]
		test_accuracy_temp = np.asarray(test_accuracy_data[i], dtype=float)[:, 2]

		plt.subplot(1, 2, 2)
		plt.plot(test_epoch_temp, test_accuracy_temp, label=plot_labels[i], color=plot_linecolors[i],
		         marker=plot_linemarkers[i], lw=plot_linewidths[i] * 2, markersize=plot_linewidths[i] * 3)
		plt.xlabel("Standard epochs")
		plt.ylabel("Testing accuracy (%)")
		plt.title(
			"Med3DResNet testing accuracy vs. epochs\n70/30 train/test (102/28), batch_size=40, initial_learning rate=0.1, 25 epochs")
		plt.legend()
		plt.ylim(0.0, 1.0)

	plt.savefig("train_test_accuracy_vs_epochs.png")
	plt.close()


if __name__ == '__main__':

	################ CONSTANTS ################
	# single_fn = "/users/timothyburt/Desktop/LIDC-IDRI-0001_normalized_3d_affine.npy"
	temp_folder = "/users/timothyburt/Desktop/video_temp"  # for images and final video
	patient_id = "0068"  # which patient should the axial_movie def. plot?
	annotations_path = "/Volumes/APPLE SSD/acv_project_team1_data/acv_image_data"  # path containing affine/projection image subfolders
	tf_ckpt_path = "/data/results"  # holds .csv exported data from TensorBoard API
	cmap = 'binary'  # see https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
	lung_mask = True
	font_size = 16  # across all plots made with this script

	make_single_movie = False
	make_comparison_movie = False
	make_tf_accuracy_plot = True
	###########################################

	plt.rcParams.update({'font.size': font_size})

	if make_comparison_movie:
		projection_fn = "%s/projection_images/LIDC-IDRI-%s_normalized_3d_projection.npy" % (
			annotations_path, patient_id)
		affine_fn = "%s/affine_images/LIDC-IDRI-%s_normalized_3d_affine.npy" % (annotations_path, patient_id)
		masked_projection_fn = "%s/projection_images/LIDC-IDRI-%s_normalized_3d_projection_masked.npy" % (
			annotations_path, patient_id)
		masked_affine_fn = "%s/affine_images/LIDC-IDRI-%s_normalized_3d_affine_masked.npy" % (
			annotations_path, patient_id)

		movie_fn = "axial_movie_PID_%s" % patient_id

		affine_img = np.load(affine_fn)
		projection_img = np.load(projection_fn)

		if lung_mask:
			masked_affine_img = np.load(masked_affine_fn)
			masked_projection_img = np.load(masked_projection_fn)
		else:
			masked_affine_img = None
			masked_projection_img = None

		if not os.path.exists(temp_folder):
			os.mkdir(temp_folder)
		os.chdir(temp_folder)

		make_axial_movie_comparison(affine_img, projection_img, masked_affine_img, masked_projection_img, cmap,
		                            movie_fn, patient_id, lung_mask=lung_mask)
	elif make_single_movie:
		single_img = np.load(single_fn)
		make_axial_movie(single_img, cmap)
	elif make_tf_accuracy_plot:
		plot_tf_accuracy(tf_ckpt_path)

	print("Done!")
