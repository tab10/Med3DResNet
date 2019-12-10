"""
preprocess
This code is depreciated, parts of it are now used directly in the annotation GUI
Author: Yuan Zi
"""

# %% import
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.ndimage import zoom

init_notebook_mode(connected=True)
# %% read patient 1
data_path = "E:/Courses/ACV/project/data/LUNA16/stage1/stage1/00edff4f51a893d80dae2d42a7f45ad1"
output_path = working_path = "E:/Courses/ACV/project/workspace/"
g = glob(data_path + '/*.dcm')

# Print out the first 5 file names to verify we're in the right folder.
print ("Total of %d DICOM images.\nFirst 5 filenames:" % len(g))
print ('\n'.join(g[:5]))
# %%read dicon as list, convert raw value to Houndsfeld units

# Loop over the image files and store everything into a list.
# 

def load_scan(path):
	slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
	slices.sort(key = lambda x: int(x.InstanceNumber))
	try:
		slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
	except:
		slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

	for s in slices:
		s.SliceThickness = slice_thickness

	return slices


def get_pixels_hu(scans):
	image = np.stack([s.pixel_array for s in scans])
	# Convert to int16 (from sometimes int16),
	# should be possible as values should always be low enough (<32k)
	image = image.astype(np.int16)

	# Set outside-of-scan pixels to 1
	# The intercept is usually -1024, so air is approximately 0
	image[image == -2000] = 0

	# Convert to Hounsfield units (HU)
	intercept = scans[0].RescaleIntercept
	slope = scans[0].RescaleSlope

	if slope != 1:
		image = slope * image.astype(np.float64)
		image = image.astype(np.int16)

	image += np.int16(intercept)

	return np.array(image, dtype=np.int16)
# %%

# %% extract crop points coordinate from jason file
import json
def read_crop_points(file_path):
	crop_points=[]
	with open(file_path) as json_file:
		data = json.load(json_file)
		for p in data['slices']:
			crop_points.append(p['bounds'])
			print('file_name'+ p['file_name'])
			print('crop_points: ' + str(p['bounds']))
			#print('num_slices: ' + p['num_slices'])
			print('')
	return crop_points#, num_superior_slice, num_inferior_slice

crop_points=read_crop_points("\users\timothyburt\Desktop\annotations.txt")

point = Point(0.5, 0.5)
polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
print(polygon.contains(point))



# %% creat mask with annotation file(each slices 4 points) and crop images.

def make_mask(crop_points,img, display=False):
	row_size = img.shape[0]
	col_size = img.shape[1]
	# Create a Polygon

	coords = crop_points
	polygon = Polygon(coords)
	mask = np.ndarray([row_size, col_size], dtype=np.int8)
	mask[:] = 0
	for i in range(row_size):
		for j in range(col_size):
			mask[i,j]=polygon.contains(Point(i, j))
	# show image before and after mask
	if (display):
		fig, ax = plt.subplots(3, 2, figsize=[12, 12])
		ax[0, 0].set_title("Original")
		ax[0, 0].imshow(img, cmap='gray')
		ax[0, 0].axis('off')
		ax[0, 1].set_title("mask")
		ax[0, 1].imshow(mask, cmap='gray')
		ax[0, 1].axis('off')
		ax[1, 0].set_title("after crop")
		ax[1, 0].imshow(mask * img, cmap='gray')
		ax[1, 0].axis('off')
		plt.show()
	return mask * img
# %%

# %%take patient 0 slice 1 as example, calculate mask and crop with mask
img = imgs[0]
crop_points = ([223, 182], [289, 206], [265, 260], [201, 242])
make_mask(crop_points,img, display=True)

# %% save patien 0 all slices' HU type as a file
id=0
patient = load_scan(data_path)
imgs = get_pixels_hu(patient)
# %%save converted images
np.save(output_path + "fullimages_%d.npy" % (id), imgs)
# %% create a histogram of all the voxel data in the study
file_used=output_path+"fullimages_%d.npy" % id
imgs_to_process = np.load(file_used).astype(np.float64)

# %% flatten all slices and show HU statistic histogram
plt.hist(imgs_to_process.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()
# %%

fig, ax = plt.subplots(3, 2, figsize=[16, 16])
ax[0, 0].set_title("HU")
ax[0, 0].imshow(imgs[0])
ax[0, 0].axis('off')
ax[0, 1].set_title("after resamp")
ax[0, 1].imshow(imgs_after_resamp[0])
ax[0, 1].axis('off')
ax[1, 0].set_title("original")
ax[1, 0].imshow(slices[0].pixel_array)
ax[1, 0].axis('off')
ax[1, 1].set_title("after max min normaliza")
ax[1, 1].imshow(imgs_after_maxmin_normaliza[0])
ax[1, 1].axis('off')
ax[2, 0].set_title("after zero center")
ax[2, 0].imshow(imgs_after_zero_center[0])
ax[2, 0].axis('off')
ax[2, 1].set_title("after resample to 256*256*256")
ax[2, 1].imshow(imgs_output[0])
ax[2, 1].axis('off')
plt.show()

# %%load patient's slices as imgs_to_process
id = 0
imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))
# %%

# %%show every 3 slices
def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
	fig,ax = plt.subplots(rows,cols,figsize=[12,12])
	for i in range(rows*cols):
		ind = start_with + i*show_every
		ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
		ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
		ax[int(i/rows),int(i % rows)].axis('off')
	plt.show()

sample_stack(imgs_to_process)
# %%

# %% print slice's thickness
print ("Slice Thickness: %f" % patient[0].SliceThickness)
print ("Pixel Spacing (row, col): (%f, %f) " % (patient[0].PixelSpacing[0], patient[0].PixelSpacing[1]))
# %%

# %%resampling(make sure each voxels represents 1*1*1mm pixels)
id = 0
imgs_to_process = np.load(output_path + 'fullimages_{}.npy'.format(id))


def resample(image, scan, new_spacing=[1, 1, 1]):
	# Determine current pixel spacing
	spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)

	resize_factor = spacing / new_spacing
	new_real_shape = image.shape * resize_factor
	new_shape = np.round(new_real_shape)
	real_resize_factor = new_shape / image.shape
	new_spacing = spacing / real_resize_factor

	image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

	return image, new_spacing
# %%

# %% print size before and after resample

print("Shape before resampling\t", imgs_to_process.shape)
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1, 1, 1])

print("Shape after resampling\t", imgs_after_resamp.shape)
# %%

# %%max min normalization
MIN_BOUND = imgs_after_resamp.min()
MAX_BOUND = imgs_after_resamp.max()
def max_min_normalize(image):
	image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
	image[image > 1] = 1.
	image[image < 0] = 0.
	return image

imgs_after_maxmin_normaliza=max_min_normalize(imgs_after_resamp)


PIXEL_MEAN = 0.4424735201862174

# %%Zero centering
def zero_center(image):
	image = image - PIXEL_MEAN
	return image

imgs_after_zero_center=zero_center(imgs_after_maxmin_normaliza)
# %%

# %%resample_cnn to fit CNN input
def resample_cnn(image, output_shape):
	new_image = zoom(image, (output_shape[0]/image.shape[0], output_shape[1]/image.shape[1], output_shape[1]/image.shape[1]))
	return new_image
imgs_output=resample_cnn(imgs_after_zero_center,[256,256,256])

# %% 3D plot
def make_mesh(image, threshold=-300, step_size=1):
	print
	("Transposing surface")
	p = image.transpose(2, 1, 0)

	print
	("Calculating surface")
	verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True)
	return verts, faces


def plotly_3d(verts, faces):
	x, y, z = zip(*verts)

	print
	"Drawing"

	# Make the colormap single color since the axes are positional not intensity.
	#    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
	colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']

	fig = FF.create_trisurf(x=x,
	                        y=y,
	                        z=z,
	                        plot_edges=False,
	                        colormap=colormap,
	                        simplices=faces,
	                        backgroundcolor='rgb(64, 64, 64)',
	                        title="Interactive Visualization")
	iplot(fig)


def plt_3d(verts, faces):
	print
	"Drawing"
	x, y, z = zip(*verts)
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111, projection='3d')

	# Fancy indexing: `verts[faces]` to generate a collection of triangles
	mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
	face_color = [1, 1, 0.9]
	mesh.set_facecolor(face_color)
	ax.add_collection3d(mesh)

	ax.set_xlim(0, max(x))
	ax.set_ylim(0, max(y))
	ax.set_zlim(0, max(z))
	ax.set_facecolor((0.7, 0.7, 0.7))
	plt.show()
# %%
def plot_3d(image, threshold=-300):
	p = image.transpose(2,1,0)
	verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111, projection='3d')
	mesh = Poly3DCollection(verts[faces], alpha=0.1)
	face_color = [0.5, 0.5, 1]
	mesh.set_facecolor(face_color)
	ax.add_collection3d(mesh)
	ax.set_xlim(0, p.shape[0])
	ax.set_ylim(0, p.shape[1])
	ax.set_zlim(0, p.shape[2])

	plt.show()
# %%show 3D image

v, f = make_mesh(imgs,-1000)
plt_3d(v, f)

v, f = make_mesh(imgs_output)
plt_3d(v, f)


v, f = make_mesh(imgs_after_resamp, 350, 2)
plotly_3d(v, f)
# %%



