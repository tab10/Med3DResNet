# this code trains and tests a 3d cnn on the hearts
# Tim Burt 11/15/19


import tensorflow as tf
import numpy as np
import glob
import os
import argparse
import csv
from visualization import *
import logging                                                     ##...for making logs on progress based on real time
from tensorflow.python.client import device_lib                    ##...to check whether GPU is called properly or not
local_device_protos = device_lib.list_local_devices()


def load_data(path, flag='affine', lungmask=True):
	"""
	INPUTS:
	:path: 
	:print_mod:
	:flag:
	OUTPUTS:
	
	"""

	train_images = []
	test_images = []
	train_classes = []
	test_classes = []

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
		train_image_temp = np.load("%s/%s_images/%s_normalized_3d_%s.npy" % (path, flag, train_labels[i][0], flag))
		masked_train_image_temp = []
		if lungmask:
			for j in range(len(train_image_temp)):
				slice = np.squeeze(train_image_temp)[:][:][j]
				slice_mask = make_lungmask(slice)
				slice_masked = apply_lungmask(slice, slice_mask)
				masked_train_image_temp.append(slice_masked)
			masked_train_image_temp += offset
			MIN_HU, MAX_HU = _update_hu_range_(masked_train_image_temp, MIN_HU, MAX_HU)
			train_images.append(masked_train_image_temp)
		else:
			train_image_temp += offset
			MIN_HU, MAX_HU = _update_hu_range_(train_image_temp, MIN_HU, MAX_HU)
			train_images.append(np.squeeze(train_image_temp))

		train_classes.append(int(train_labels[i][1]))
	for i in range(len(test_labels)):
		test_image_temp = np.load(
			"%s/%s_images/%s_normalized_3d_%s.npy" % (path, flag, test_labels[i][0], flag))
		masked_test_image_temp = []
		if lungmask:
			for j in range(len(test_image_temp)):
				slice = np.squeeze(test_image_temp)[:][:][j]
				slice_mask = make_lungmask(slice)
				slice_masked = apply_lungmask(slice, slice_mask)
				masked_test_image_temp.append(slice_masked)
			masked_test_image_temp += offset
			MIN_HU, MAX_HU = _update_hu_range_(masked_test_image_temp, MIN_HU, MAX_HU)
			test_images.append(masked_test_image_temp)
		else:
			test_image_temp += offset
			MIN_HU, MAX_HU = _update_hu_range_(test_image_temp, MIN_HU, MAX_HU)
			test_images.append(np.squeeze(test_image_temp))
		test_classes.append(int(test_labels[i][1]))

	n_channels = int(MAX_HU - MIN_HU + 1)
	print("Global min HU: %d, global max HU: %d. Input image channels: %d" % (MIN_HU, MAX_HU, n_channels))

	return train_images, test_images, train_classes, test_classes, n_channels


def conv3d(x, W):
	return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
	#                        size of window         movement of window as you slide about
	return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def build_cnn(x, n_classes, n_channels, features_root=32):
	#                # 3 x 3 x 3 patches, foo channel, bar features to compute.
	weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, n_channels, features_root])),
			   'W_conv2':  tf.Variable(tf.random_normal([3, 3, 3, n_channels, features_root*2])),
			   'W_conv3':  tf.Variable(tf.random_normal([3, 3, 3, n_channels, features_root*3])),
			   'W_conv4':  tf.Variable(tf.random_normal([3, 3, 3, n_channels, features_root*4])),
			   'W_conv5':  tf.Variable(tf.random_normal([3, 3, 3, n_channels, features_root*5])),
			   'W_conv6':  tf.Variable(tf.random_normal([3, 3, 3, n_channels, features_root*6])),
			   'W_conv7':  tf.Variable(tf.random_normal([3, 3, 3, n_channels, features_root*7])),
			   'W_conv8':  tf.Variable(tf.random_normal([3, 3, 3, n_channels, features_root*8])),
			   'W_fc': tf.Variable(tf.random_normal([54080, 64*n_channels])),
			   'out': tf.Variable(tf.random_normal([1024, n_classes]))}

	biases = {'b_conv1': tf.Variable(tf.random_normal([features_root])),
			  'b_conv2': tf.Variable(tf.random_normal([features_root*2])),
			  'b_conv3': tf.Variable(tf.random_normal([features_root * 3])),
			  'b_conv4': tf.Variable(tf.random_normal([features_root * 4])),
			  'b_conv5': tf.Variable(tf.random_normal([features_root * 5])),
			  'b_conv6': tf.Variable(tf.random_normal([features_root * 6])),
			  'b_conv7': tf.Variable(tf.random_normal([features_root * 7])),
			  'b_conv8': tf.Variable(tf.random_normal([features_root * 8])),
			  'b_fc': tf.Variable(tf.random_normal([64*n_channels])),
			  'out': tf.Variable(tf.random_normal([n_classes]))}

	#                            image X      image Y        image Z
	x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

	conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
	conv1 = maxpool3d(conv1)

	conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
	conv2 = maxpool3d(conv2)

	fc = tf.reshape(conv2, [-1, 54080])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
	fc = tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc, weights['out']) + biases['out']

	return output


def train_cnn(train_data, train_classes, n_channels):
	prediction = build_cnn(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_classes, logits=prediction))
	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

	hm_epochs = 10
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		successful_runs = 0
		total_runs = 0

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for data in train_data:
				total_runs += 1
				try:
					X = data[0]
					Y = data[1]
					_, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
					epoch_loss += c
					successful_runs += 1
				except Exception as e:
					# I am passing for the sake of notebook space, but we are getting 1 shaping issue from one
					# input tensor. Not sure why, will have to look into it. Guessing it's
					# one of the depths that doesn't come to 20.
					pass
			# print(str(e))

			print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

			print('Accuracy:',
				  accuracy.eval({x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]}))

		print('Done. Finishing accuracy:')
		print('Accuracy:', accuracy.eval({x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]}))

		print('fitment percent:', successful_runs / total_runs)


if __name__ == "__main__":

	################ CONSTANTS ################
	IMG_SIZE_PX = 256
	SLICE_COUNT = 256
	n_classes = 4
	batch_size = 157  # total number of ct data arrays, for nn normalization

	work_path = "/Volumes/APPLE SSD"
	data_path = "acv_image_data"  # subfolder from ran dir where images are stored
	flag = 'affine'  # train CNN on these images
	keep_rate = 0.8
	###########################################

	os.chdir(work_path)

	x = tf.placeholder('float')
	y = tf.placeholder('float')

	input_dims = [IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT]

	train_data, test_data, train_classes, test_classes, n_channels = load_data(data_path, flag=flag, lungmask=False)

	train_cnn(train_data, train_classes, n_channels)
