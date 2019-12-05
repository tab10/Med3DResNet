# this code trains and tests a 3d cnn on the hearts
# Tim Burt 11/15/19


import tensorflow as tf
import numpy as np
import glob
import os
import argparse


def load_data(path, print_mod=10, flag='affine'):
	"""
	INPUTS:
	:path: 
	:print_mod:
	:flag:
	OUTPUTS:
	
	"""

	train_images = []
	test_images = []

	print("Loading diagnostic truth class labels...")  # these are the diagnostic truth
	
	train_labels = np.loadtxt("acv_train_labels.csv", skiprows=1, usecols=(0, 1))
	test_labels = np.loadtxt("acv_test_labels.csv", skiprows=1, usecols=(0, 1))

	# sort image data into train/test data according to keys from .csv files
	print("Loading/sorting image data into test/train split from .csv files...")
	print("Loading %s images..." % flag)
	for i in range(len(train_labels[0])):
		if i % int(len(train_labels[0]) / print_mod) == 0:
			print("Loading train image %d of %d..." % (i + 1, len(train_labels[0])))
		train_image_temp = np.load("%s/%s_images/%s_normalized_3d_%s.npy" % (path, flag, train_labels[0][i], flag))
		train_images.append(np.squeeze(train_image_temp))
	for i in range(len(test_labels[0])):
		if i % int(len(test_labels[0]) / print_mod) == 0:
			print("Loading test image %d of %d..." % (i + 1, len(test_labels[0])))
		test_image_temp = np.load(
			"%s/%s_images/%s_normalized_3d_%s.npy" % (path, flag, test_labels[0][i], flag))
		test_images.append(np.squeeze(test_image_temp))

	train_classes = train_labels[1]
	test_classes = test_labels[1]

	return train_images, test_images, train_classes, test_classes


def conv3d(x, W):
	return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
	#                        size of window         movement of window as you slide about
	return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def build_cnn(x):
	#                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
	weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
			   # 5 x 5 x 5 patches, 32 channels, 64 features to compute.
			   'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),  # 64 features
			   'W_fc': tf.Variable(tf.random_normal([54080, 1024])),
			   'out': tf.Variable(tf.random_normal([1024, n_classes]))}

	biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
			  'b_conv2': tf.Variable(tf.random_normal([64])),
			  'b_fc': tf.Variable(tf.random_normal([1024])),
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


def train_cnn(x):
	prediction = build_cnn(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

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
	data_path = "acv_image_data"  # subfolder from ran dir where images are stored
	flag = 'affine'  # train CNN on these images
	keep_rate = 0.8
	###########################################

	x = tf.placeholder('float')
	y = tf.placeholder('float')

	input_dims = [IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT]

	train_data, validation_data, train_classes, test_classes = load_data(data_path, flag=flag)

	train_cnn(x)
