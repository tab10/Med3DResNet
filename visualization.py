# visualization defs go here
# tim burt 11/30/19

import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob


def make_axial_movie(image, cmap, movie_fn="axial_movie_norm_affine", fps=8):
	num_slices = image.shape[0]  # assuming square
	slices = np.squeeze(image)
	# find global voxel max/min and set colorbar to that fixed range
	max_all = np.max(np.max(slices))
	min_all = np.min(np.min(slices))
	for i in range(num_slices):
		if (i % int(num_slices / 10)) == 0:
			print("Plotting slice %d of %d..." % (i+1, num_slices))
		slice = slices[:][:][i]
		plt.imshow(slice, cmap=plt.get_cmap(cmap), vmin=min_all+1, vmax=max_all)  # +1 removes bkgd intensity
		plt.colorbar()
		plt.title("Affine Method\nAxial slice %d of %d\nHounsfield Units" % (i+1, num_slices))
		plt.xlabel("X (pixels)")
		plt.ylabel("Y (pixels)")
		plt.savefig("slice_%d.png" % i)
		plt.close()

	print("Creating movie...")
	os.system("ffmpeg -r %d -i slice_%%d.png -r 30 %s.mp4" % (fps, movie_fn))


def make_axial_movie_comparison(affine_image, norm_affine_image, cmap, movie_fn="axial_movie_comparison_tab20b_cmap_scaled", fps=8):
	num_slices = affine_image.shape[0]  # assuming square and both images same size
	
	affine_slices = np.squeeze(affine_image)
	# find affine_image global voxel max/min and set colorbar to that fixed range
	affine_max_all = np.max(np.max(affine_slices))
	affine_min_all = np.min(np.min(affine_slices))
	
	norm_affine_slices = np.squeeze(norm_affine_image)
	# find norm_affine_image global voxel max/min and set colorbar to that fixed range
	norm_affine_max_all = np.max(np.max(norm_affine_slices))
	norm_affine_min_all = np.min(np.min(norm_affine_slices))

	for i in range(num_slices):
		if (i % int(num_slices / 10)) == 0:
			print("Plotting slice %d of %d..." % (i+1, num_slices))

		affine_slice = affine_slices[:][:][i]
		norm_affine_slice = norm_affine_slices[:][:][i]

		# affine plot on left
		plt.figure(figsize=(16,9))
		plt.subplot(1,2,1)
		plt.imshow(affine_slice, cmap=plt.get_cmap(cmap), vmin=affine_min_all, vmax=affine_max_all)
		plt.colorbar()
		plt.title("Projection Method\nAxial slice %d of %d\nHounsfield Units" % (i+1, num_slices))
		plt.xlabel("X (pixels)")
		plt.ylabel("Y (pixels)")

		# normalized affine plot on right
		plt.subplot(1, 2, 2)
		plt.imshow(norm_affine_slice, cmap=plt.get_cmap(cmap), vmin=norm_affine_min_all, vmax=norm_affine_max_all)
		plt.colorbar()
		plt.title("Affine Method\nAxial slice %d of %d\nHounsfield Units" % (i+1, num_slices))
		plt.xlabel("X (pixels)")
		plt.ylabel("Y (pixels)")

		#plt.tight_layout()
		plt.savefig("slice_%d.png" % i)
		plt.close()

	print("Creating movie...")
	os.system("ffmpeg -r %d -i slice_%%d.png -r 30 %s.mp4" % (fps, movie_fn))

if __name__ == '__main__':

	single_fn = "/users/timothyburt/Desktop/LIDC-IDRI-0001_normalized_3d_affine.npy"
	temp_folder = "/users/timothyburt/Desktop/video_temp"

	affine_fn = "/Users/timothyburt/Desktop/LIDC-IDRI-0001_normalized_3d_projection.npy"
	norm_affine_fn = "/Users/timothyburt/Desktop/LIDC-IDRI-0001_normalized_3d_affine.npy"

	# see https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
	cmap = 'binary'
	#cmap = 'coolwarm'

	single_img = np.load(single_fn)
	#affine_img = np.load(affine_fn)
	#norm_affine_img = np.load(norm_affine_fn)

	if not os.path.exists(temp_folder):
		os.mkdir(temp_folder)
	os.chdir(temp_folder)

	make_axial_movie(single_img, cmap)
	#make_axial_movie_comparison(affine_img, norm_affine_img, cmap)
	print("Done!")
