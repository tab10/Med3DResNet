# visualization defs go here
# tim burt 11/30/19

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


if __name__ == '__main__':
	fn = "/users/timothyburt/Desktop/LIDC-IDRI-0001_cropped3d.npy"

	# %%show 3D image
	imgs_raw = np.load(fn)
	imgs = np.squeeze(imgs_raw, axis=-1)  # removes last axis of intensity from data

	v, f = make_mesh(imgs, 100)
	plt_3d(v, f)
	plt.close()




