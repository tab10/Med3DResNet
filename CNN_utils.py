# based on package from https://github.com/taki0112/ResNet-Tensorflow
# Tim Burt 12/8/19

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from keras.datasets import cifar10, cifar100, mnist, fashion_mnist
from keras.utils import to_categorical
import numpy as np
import random
from scipy import misc
import csv
from visualization import *


def _update_hu_range_(img, cur_min, cur_max):
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

def load_cifar10() :
	(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
	# train_data = train_data / 255.0
	# test_data = test_data / 255.0

	train_data, test_data = normalize(train_data, test_data)

	train_labels = to_categorical(train_labels, 10)
	test_labels = to_categorical(test_labels, 10)

	seed = 777
	np.random.seed(seed)
	np.random.shuffle(train_data)
	np.random.seed(seed)
	np.random.shuffle(train_labels)


	return train_data, train_labels, test_data, test_labels

def load_cifar100() :
	(train_data, train_labels), (test_data, test_labels) = cifar100.load_data()
	# train_data = train_data / 255.0
	# test_data = test_data / 255.0
	train_data, test_data = normalize(train_data, test_data)

	train_labels = to_categorical(train_labels, 100)
	test_labels = to_categorical(test_labels, 100)

	seed = 777
	np.random.seed(seed)
	np.random.shuffle(train_data)
	np.random.seed(seed)
	np.random.shuffle(train_labels)


	return train_data, train_labels, test_data, test_labels

def load_mnist() :
	(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
	train_data = np.expand_dims(train_data, axis=-1)
	test_data = np.expand_dims(test_data, axis=-1)

	train_data, test_data = normalize(train_data, test_data)

	train_labels = to_categorical(train_labels, 10)
	test_labels = to_categorical(test_labels, 10)

	seed = 777
	np.random.seed(seed)
	np.random.shuffle(train_data)
	np.random.seed(seed)
	np.random.shuffle(train_labels)


	return train_data, train_labels, test_data, test_labels

def load_fashion() :
	(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
	train_data = np.expand_dims(train_data, axis=-1)
	test_data = np.expand_dims(test_data, axis=-1)

	train_data, test_data = normalize(train_data, test_data)

	train_labels = to_categorical(train_labels, 10)
	test_labels = to_categorical(test_labels, 10)

	seed = 777
	np.random.seed(seed)
	np.random.shuffle(train_data)
	np.random.seed(seed)
	np.random.shuffle(train_labels)


	return train_data, train_labels, test_data, test_labels

def load_tiny() :
	IMAGENET_MEAN = [123.68, 116.78, 103.94]
	path = './tiny-imagenet-200'
	num_classes = 200

	print('Loading ' + str(num_classes) + ' classes')

	X_train = np.zeros([num_classes * 500, 3, 64, 64], dtype=np.float32)
	y_train = np.zeros([num_classes * 500], dtype=np.float32)

	trainPath = path + '/train'

	print('loading training images...')

	i = 0
	j = 0
	annotations = {}
	for sChild in os.listdir(trainPath):
		sChildPath = os.path.join(os.path.join(trainPath, sChild), 'images')
		annotations[sChild] = j
		for c in os.listdir(sChildPath):
			X = misc.imread(os.path.join(sChildPath, c), mode='RGB')
			if len(np.shape(X)) == 2:
				X_train[i] = np.array([X, X, X])
			else:
				X_train[i] = np.transpose(X, (2, 0, 1))
			y_train[i] = j
			i += 1
		j += 1
		if (j >= num_classes):
			break

	print('finished loading training images')

	val_annotations_map = get_annotations_map()

	X_test = np.zeros([num_classes * 50, 3, 64, 64], dtype=np.float32)
	y_test = np.zeros([num_classes * 50], dtype=np.float32)

	print('loading test images...')

	i = 0
	testPath = path + '/val/images'
	for sChild in os.listdir(testPath):
		if val_annotations_map[sChild] in annotations.keys():
			sChildPath = os.path.join(testPath, sChild)
			X = misc.imread(sChildPath, mode='RGB')
			if len(np.shape(X)) == 2:
				X_test[i] = np.array([X, X, X])
			else:
				X_test[i] = np.transpose(X, (2, 0, 1))
			y_test[i] = annotations[val_annotations_map[sChild]]
			i += 1
		else:
			pass

	print('finished loading test images : ' + str(i))

	X_train = X_train.astype(np.float32)
	X_test = X_test.astype(np.float32)
	# X_train /= 255.0
	# X_test /= 255.0

	# for i in range(3) :
	#     X_train[:, :, :, i] =  X_train[:, :, :, i] - IMAGENET_MEAN[i]
	#     X_test[:, :, :, i] = X_test[:, :, :, i] - IMAGENET_MEAN[i]

	X_train, X_test = normalize(X_train, X_test)


	# convert class vectors to binary class matrices
	y_train = to_categorical(y_train, num_classes)
	y_test = to_categorical(y_test, num_classes)

	X_train = np.transpose(X_train, [0, 3, 2, 1])
	X_test = np.transpose(X_test, [0, 3, 2, 1])

	seed = 777
	np.random.seed(seed)
	np.random.shuffle(X_train)
	np.random.seed(seed)
	np.random.shuffle(y_train)

	return X_train, y_train, X_test, y_test


def load_ACV(data_folder, flag='affine', lungmask=True):
	"""
	INPUTS:
	:data_folder:
	:print_mod:
	:flag:
	OUTPUTS:

	"""

	train_data = []
	test_data = []

	train_labels = []
	test_labels = []

	offset = 2048  # Offsetting the grayscale values by 2048 HU units to make all values positive. (int16)

	MIN_HU = 0.0  # these will give the max/min range of HU intensity values (ints) which determines CNN input image channels
	MAX_HU = 0.0

	print("Loading diagnostic truth class labels...")  # these are the diagnostic truth

	with open("acv_train_labels.csv", "r") as f:
		reader = csv.reader(f)
		train_labels = list(reader)[1:]
	f.close()

	with open("acv_test_labels.csv", "r") as f:
		reader = csv.reader(f)
		test_labels = list(reader)[1:]
	f.close()

	# sort image data into train/test data according to keys from .csv files
	print("Loading/sorting image data into test/train split from .csv files...")
	print("Loading %s images..." % flag)
	if lungmask:
		print("Creating erosion/dilation masks to remove lungs...")
	for i in range(len(train_labels)):
		train_image_temp = np.load("%s/%s_images/%s_normalized_3d_%s.npy" % (data_folder, flag, train_labels[i][0], flag))
		masked_train_image_temp = []
		if lungmask:
			for j in range(len(train_image_temp)):
				slice = np.squeeze(train_image_temp)[:][:][j]
				slice_mask = make_lungmask(slice)
				slice_masked = apply_lungmask(slice, slice_mask)
				masked_train_image_temp.append(slice_masked)
			masked_train_image_temp += offset
			MIN_HU, MAX_HU = _update_hu_range_(masked_train_image_temp, MIN_HU, MAX_HU)
			train_data.append(masked_train_image_temp)
		else:
			train_image_temp += offset
			MIN_HU, MAX_HU = _update_hu_range_(train_image_temp, MIN_HU, MAX_HU)
			train_data.append(np.squeeze(train_image_temp))
		train_labels.append(int(train_labels[i][1]))
	for i in range(len(test_labels)):
		test_image_temp = np.load(
			"%s/%s_images/%s_normalized_3d_%s.npy" % (data_folder, flag, test_labels[i][0], flag))
		masked_test_image_temp = []
		if lungmask:
			for j in range(len(test_image_temp)):
				slice = np.squeeze(test_image_temp)[:][:][j]
				slice_mask = make_lungmask(slice)
				slice_masked = apply_lungmask(slice, slice_mask)
				masked_test_image_temp.append(slice_masked)
			masked_test_image_temp += offset
			MIN_HU, MAX_HU = _update_hu_range_(masked_test_image_temp, MIN_HU, MAX_HU)
			test_data.append(masked_test_image_temp)
		else:
			test_image_temp += offset
			MIN_HU, MAX_HU = _update_hu_range_(test_image_temp, MIN_HU, MAX_HU)
			test_data.append(np.squeeze(test_image_temp))
		test_labels.append(int(test_labels[i][1]))

	n_channels = int(MAX_HU - MIN_HU + 1)
	print("Global min HU: %d, global max HU: %d. Input image channels: 1" % (MIN_HU, MAX_HU))

	train_data = np.expand_dims(train_data, axis=-1)
	test_data = np.expand_dims(test_data, axis=-1)

	train_data, test_data = normalize(train_data, test_data)

	#train_labels = to_categorical(train_labels, 4)
	#test_labels = to_categorical(test_labels, 4)

	seed = 777
	np.random.seed(seed)
	np.random.shuffle(train_data)
	np.random.seed(seed)
	np.random.shuffle(train_labels)

	return train_data, train_labels, test_data, test_labels


def normalize(X_train, X_test):

	mean = np.mean(X_train, axis=(0, 1, 2, 3, 4))
	std = np.std(X_train, axis=(0, 1, 2, 3, 4))

	X_train = (X_train - mean) / std
	X_test = (X_test - mean) / std

	return X_train, X_test


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
	if dataset_name == 'mnist' :
		batch = _random_crop(batch, [img_size, img_size], 4)

	elif dataset_name =='tiny' :
		batch = _random_flip_leftright(batch)
		batch = _random_crop(batch, [img_size, img_size], 8)

	else :
		batch = _random_flip_leftright(batch)
		batch = _random_crop(batch, [img_size, img_size], 4)
	return batch