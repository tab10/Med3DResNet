# visualization defs go here
# tim burt 11/30/19

import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob


def make_axial_movie(image, movie_fn="axial_movie", fps=8):
	num_slices = image.shape[0]  # assuming square
	slices = np.squeeze(image)
	# find global voxel max/min and set colorbar to that fixed range
	max_all = np.max(np.max(slices))
	min_all = np.min(np.min(slices))
	for i in range(num_slices):
		if (i % int(num_slices / 10)) == 0:
			print("Plotting slice %d of %d..." % (i+1, num_slices))
		slice = slices[:][:][i]
		plt.imshow(slice, vmin=min_all, vmax=max_all)
		plt.colorbar()
		plt.title("Axial slice %d of %d\nHounsfield Units" % (i, image.shape[0]))
		plt.xlabel("X (pixels)")
		plt.ylabel("Y (pixels)")
		plt.savefig("slice_%d.png" % i)
		plt.close()

	print("Creating movie...")
	os.system("ffmpeg -r %d -i slice_%%d.png -r 30 %s.mp4" % (fps, movie_fn))


if __name__ == '__main__':
	fn = "/users/timothyburt/Desktop/LIDC-IDRI-0001_cropped3d.npy"
	temp_folder = "/users/timothyburt/Desktop/video_temp"

	imgs_raw = np.load(fn)
	if not os.path.exists(temp_folder):
		os.mkdir(temp_folder)
	os.chdir(temp_folder)

	make_axial_movie(imgs_raw)
	print("Done!")
