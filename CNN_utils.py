# based on package from https://github.com/taki0112/ResNet-Tensorflow
# Tim Burt 12/8/19

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf
import tensorflow.contrib.slim as slim

# append to lines above
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

from keras.datasets import cifar10, cifar100, mnist, fashion_mnist
from keras.utils import to_categorical
import numpy as np

import random
from scipy import misc
import csv
from visualization import *

# append to lines above
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}


def update_hu_range(img, cur_min, cur_max):
	local_min_hu = np.amin(img)
	local_max_hu = np.amax(img)
	if local_min_hu < cur_min:
		cur_min = local_min_hu
	if local_max_hu > cur_max:
		cur_max = local_max_hu
	return cur_min, cur_max


def check_folder(log_dir):
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	return log_dir

def show_all_variables():
	model_vars = tf.trainable_variables()
	slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
	return x.lower() in ('true')


def load_ACV(data_folder, n_axial_channels, flag, use_lung_mask, train_test_ratio, verbose=False):
	"""
	"""

	train_data = []
	test_data = []

	train_labels_list = []
	test_labels_list = []

	offset = 2048  # Offsetting the grayscale values by 2048 HU units to make all values positive. (int16)

	MIN_HU = 0.0  # these will give the max/min range of HU intensity values (ints) which determines CNN input image channels
	MAX_HU = 0.0

	print("Loading diagnostic truth class labels...")  # these are the diagnostic truth

	if train_test_ratio == '70_30':
		print("%s train/test ratio being used..." % train_test_ratio)
		train_labels_fn = "acv_train_labels_%s_3_class.csv" % train_test_ratio
		test_labels_fn = "acv_test_labels_%s_3_class.csv" % train_test_ratio
	elif train_test_ratio == '80_20':
		print("%s train/test ratio being used..." % train_test_ratio)
		train_labels_fn = "acv_train_labels_%s_3_class.csv" % train_test_ratio
		test_labels_fn = "acv_test_labels_%s_3_class.csv" % train_test_ratio
	else:
		print("Check train/test ratio value!")
		raise SystemExit

	with open(train_labels_fn, "r") as f:
		reader = csv.reader(f)
		train_labels = list(reader)[1:]
	f.close()

	with open(test_labels_fn, "r") as f:
		reader = csv.reader(f)
		test_labels = list(reader)[1:]
	f.close()

	# sort image data into train/test data according to keys from .csv files
	print("Loading/sorting image data into test/train split from .csv files...")
	print("Loading %s images..." % flag)
	if use_lung_mask:
		print("Loading erosion/dilation masked images...")
	for i in range(len(train_labels)):
		if use_lung_mask:
			train_image_fn = "%s/%s_images/%s_normalized_3d_%s_masked.npy" % (data_folder, flag, train_labels[i][0], flag)
		else:
			train_image_fn = "%s/%s_images/%s_normalized_3d_%s.npy" % (data_folder, flag, train_labels[i][0], flag)
		train_image_temp = np.load(train_image_fn)
		train_image_temp += offset
		MIN_HU, MAX_HU = update_hu_range(train_image_temp, MIN_HU, MAX_HU)
		train_data.append(np.squeeze(train_image_temp))
		train_labels_list.append(int(train_labels[i][1]))

	for i in range(len(test_labels)):
		if use_lung_mask:
			test_image_fn = "%s/%s_images/%s_normalized_3d_%s_masked.npy" % (data_folder, flag, test_labels[i][0], flag)
		else:
			test_image_fn = "%s/%s_images/%s_normalized_3d_%s.npy" % (data_folder, flag, test_labels[i][0], flag)
		test_image_temp = np.load(test_image_fn)
		test_image_temp += offset
		MIN_HU, MAX_HU = update_hu_range(test_image_temp, MIN_HU, MAX_HU)
		test_data.append(np.squeeze(test_image_temp))
		test_labels_list.append(int(test_labels[i][1]))

	train_data, test_data = normalize(train_data, test_data, n_axial_channels, verbose=verbose)

	# class label gymnastics
	train_labels = np.asarray(train_labels_list) - 1  # -1 moves classes to [0,1,2], excluding unknown class
	test_labels = np.asarray(test_labels_list) - 1

	if verbose:
		print("Global min HU: %d, global max HU: %d before [0,1] map. Input image channels: %d" % (
		MIN_HU, MAX_HU, n_axial_channels))
		print(train_labels.shape, train_data.shape)

	train_labels = to_categorical(train_labels, 3)
	test_labels = to_categorical(test_labels, 3)

	seed = 777
	np.random.seed(seed)
	np.random.shuffle(train_data)
	np.random.seed(seed)
	np.random.shuffle(train_labels)

	return train_data, train_labels, test_data, test_labels


def normalize(X_train, X_test, n_axial_channels, verbose=False):

	X_train_z_slices = []
	X_test_z_slices = []

	slices = int(256 / n_axial_channels)
	slice_width = int(256 / slices)

	for i in range(1, slices+1):
		print("Averaging z-block %d of %d..." % (i, slices))

		z_slice_start = ((i-1) * slice_width)
		z_slice_stop = (i * slice_width) - 1

		X_train_z_slice = np.mean(np.asarray(X_train)[:,:,:,z_slice_start:z_slice_stop], axis=3)
		X_test_z_slice = np.mean(np.asarray(X_test)[:,:,:,z_slice_start:z_slice_stop], axis=3)

		if verbose:
			print(np.asarray(X_train).shape)
			print(np.asarray(X_train)[:, :, :, z_slice_start:z_slice_stop].shape)
			print(X_train_z_slice.shape)

		X_train_z_slices.append(X_train_z_slice)
		X_test_z_slices.append(X_test_z_slice)

	print("Normalizing z-blocks...")

	#X_train_z_slice = np.expand_dims(X_train_z_slice, axis=-1)  # for single channel images, use this
	#X_test_z_slice = np.expand_dims(X_test_z_slice, axis=-1)

	X_train_z_slices = np.transpose(np.asarray(X_train_z_slices), (1, 2, 3, 0))  # option permutes the order to (patient, x, y, channel (z-avg)
	X_test_z_slices = np.transpose(np.asarray(X_test_z_slices), (1, 2, 3, 0))

	mean = np.mean(X_train_z_slices, axis=(0, 1, 2, 3))
	std = np.std(X_train_z_slices, axis=(0, 1, 2, 3))

	X_train_z_slices = (X_train_z_slices - mean) / std
	X_test_z_slices = (X_test_z_slices - mean) / std

	print(X_train_z_slices.shape, X_test_z_slices.shape)

	return X_train_z_slices, X_test_z_slices


def get_annotations_map():
	valAnnotationsPath = './tiny-imagenet-200/val/val_annotations.txt'
	valAnnotationsFile = open(valAnnotationsPath, 'r')
	valAnnotationsContents = valAnnotationsFile.read()
	valAnnotations = {}

	for line in valAnnotationsContents.splitlines():
		pieces = line.strip().split()
		valAnnotations[pieces[0]] = pieces[1]

	return valAnnotations


def _random_crop(batch, crop_shape, padding=None):
	oshape = np.shape(batch[0])

	if padding:
		oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
	new_batch = []
	npad = ((padding, padding), (padding, padding), (0, 0))
	for i in range(len(batch)):
		new_batch.append(batch[i])
		if padding:
			new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
			                          mode='constant', constant_values=0)
		nh = random.randint(0, oshape[0] - crop_shape[0])
		nw = random.randint(0, oshape[1] - crop_shape[1])
		new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
		               nw:nw + crop_shape[1]]
	return new_batch


def _random_flip_leftright(batch):
	for i in range(len(batch)):
		if bool(random.getrandbits(1)):
			batch[i] = np.fliplr(batch[i])
	return batch


def data_augmentation(batch, img_size, dataset_name):
	batch = _random_flip_leftright(batch)
	batch = _random_crop(batch, [img_size, img_size], 4)
	return batch