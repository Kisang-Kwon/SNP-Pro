import tensorflow as tf
import numpy as np



def SNPPro(X, batch_size, dropout_rate=0.3, training=False):
	## Model structure
	x_reshape = tf.reshape(X, [batch_size, 1, 1980, 11])
	x_cnn = tf.transpose(x_reshape, perm=[0, 2, 3, 1])
	x_dnn = tf.reshape(x_cnn, [batch_size, -1])
	
	## CNN
	conv1_W = tf.Variable(tf.random_normal([15, 15, 1, 300], stddev=0.01))
	conv1 = tf.nn.conv2d(x_cnn, conv1_W, strides=[1, 1, 1, 1], padding='SAME')
	conv1 = tf.nn.relu(conv1)
	conv1 = tf.nn.max_pool(conv1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

	conv2_W = tf.Variable(tf.random_normal([7, 7, 300, 300], stddev=0.01))
	conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME')
	conv2 = tf.nn.relu(conv2)
	conv2 = tf.nn.max_pool(conv2, ksize=[1, 11, 11, 1], strides=[1, 11, 11, 1], padding='SAME')

	## DNN
	dnn1_W = tf.Variable(tf.random_normal([1980 * 11, 1980], stddev=0.01))
	dnn1 = tf.matmul(x_dnn, dnn1_W)
	dnn1 = tf.nn.relu(dnn1)
    
	if training:
		dnn1 = tf.nn.dropout(dnn1, dropout_rate)

	dnn2_W = tf.Variable(tf.random_normal([1980, int(1980 / 11)], stddev=0.01))
	dnn2 = tf.matmul(dnn1, dnn2_W)
	dnn2 = tf.nn.relu(dnn2)
	
	if training:
		dnn2 = tf.nn.dropout(dnn2, dropout_rate)

	dnn3_W = tf.Variable(tf.random_normal([int(1980 / 11), int(1980 / 22)], stddev=0.01))
	dnn3 = tf.matmul(dnn2, dnn3_W)
	dnn3 = tf.nn.relu(dnn3)
	
	if training:
		dnn3 = tf.nn.dropout(dnn3, dropout_rate)

	## FCN
	conv2_reshape = tf.reshape(conv2, [batch_size, -1])
	fc = tf.concat([conv2_reshape, dnn3], axis=1)
	
	n_nodes = int(np.shape(fc)[1])

	fc_W = tf.Variable(tf.random_normal([n_nodes, int(n_nodes/100)], stddev=0.01))
	fc1 = tf.matmul(fc, fc_W)
	fc1 = tf.nn.relu(fc1)
	
	if training:
		fc1 = tf.nn.dropout(fc1, dropout_rate)

	fc2_W = tf.Variable(tf.random_normal([int(n_nodes/100), int(n_nodes/1000)], stddev=0.01))
	fc2 = tf.matmul(fc1, fc2_W)
	fc2 = tf.nn.relu(fc2)
    
	if training:
		fc2 = tf.nn.dropout(fc2, dropout_rate)

	output_W = tf.Variable(tf.random_normal([int(n_nodes/1000), 1], stddev=0.01))
	output = tf.matmul(fc2, output_W)
	output = tf.nn.relu(output)
	
	return output
