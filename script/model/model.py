import tensorflow as tf
import numpy as np



def SNPPro(X, batch_size, dropout_rate, training=False):
	## Model structure
	X_reshape = tf.reshape(X, [batch_size, 1, 1980, 11])
	X_cnn = tf.transpose(X_reshape, perm=[0, 2, 3, 1])
	X_dnn = tf.reshape(X_cnn, [batch_size, -1])
	
	## CNN
	CL1_W = tf.Variable(tf.random_normal([15, 15, 1, 300], stddev=0.01))
	CL1 = tf.nn.conv2d(X_cnn, CL1_W, strides=[1, 1, 1, 1], padding='SAME')
	CL1 = tf.nn.relu(CL1)
	CL1 = tf.nn.max_pool(CL1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

	CL2_W = tf.Variable(tf.random_normal([7, 7, 300, 300], stddev=0.01))
	CL2 = tf.nn.conv2d(CL1, CL2_W, strides=[1, 1, 1, 1], padding='SAME')
	CL2 = tf.nn.relu(CL2)
	CL2 = tf.nn.max_pool(CL2, ksize=[1, 11, 11, 1], strides=[1, 11, 11, 1], padding='SAME')

	## DNN
	NL1_W = tf.Variable(tf.random_normal([1980 * 11, 1980], stddev=0.01))
	NL1 = tf.matmul(X_dnn, NL1_W)
	NL1 = tf.nn.relu(NL1)
    
	if training:
		NL1 = tf.nn.dropout(NL1, dropout_rate)

	NL2_W = tf.Variable(tf.random_normal([1980, int(1980 / 11)], stddev=0.01))
	NL2 = tf.matmul(NL1, NL2_W)
	NL2 = tf.nn.relu(NL2)
	
	if training:
		NL2 = tf.nn.dropout(NL2, dropout_rate)

	NL3_W = tf.Variable(tf.random_normal([int(1980 / 11), int(1980 / 22)], stddev=0.01))
	NL3 = tf.matmul(NL2, NL3_W)
	NL3 = tf.nn.relu(NL3)
	
	if training:
		NL3 = tf.nn.dropout(NL3, dropout_rate)

	## FCN
	CL2_reshape = tf.reshape(CL2, [batch_size, -1])
	FCL_I = tf.concat([CL2_reshape, NL3], axis=1)
	FCL_I = tf.reshape(FCL_I, [1,-1])

	tmp_node = str(np.shape(FCL_I))
	ar_node = tmp_node.split(', ')
	for_node = ar_node[1].split(')')
	length_of_input = int(for_node[0])

	FCL_W = tf.Variable(tf.random_normal([length_of_input, int(length_of_input/100)], stddev=0.01))
	FCL1 = tf.matmul(FCL_I, FCL_W)
	FCL1 = tf.nn.relu(FCL1)
	
	if training:
		FCL1 = tf.nn.dropout(FCL1, dropout_rate)

	FCL2_W = tf.Variable(tf.random_normal([int(length_of_input/100), int(length_of_input/1000)], stddev=0.01))
	FCL2 = tf.matmul(FCL1, FCL2_W)
	FCL2 = tf.nn.relu(FCL2)
    
	if training:
		FCL2 = tf.nn.dropout(FCL2, dropout_rate)

	output_W = tf.Variable(tf.random_normal([int(length_of_input/1000), batch_size], stddev=0.01))
	output = tf.matmul(FCL2, output_W)
	output = tf.nn.relu(output)
	
	return output