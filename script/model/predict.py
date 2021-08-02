import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import time

from prepare_dataset import set_data_list, get_batch_data
from dircheck import dircheck
from model import SNPPro

## File input
def get_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', required=True)
	parser.add_argument('-d', '--data', required=True)
	parser.add_argument('-v', '--version', required=True, type=str)
	parser.add_argument('-r', '--restore', action='store_true')
	
	return parser.parse_args()


if __name__ == '__main__':
	args = get_arguments()

	ckpt_dir = os.path.join(args.version, 'ckpt')
	dircheck(ckpt_dir)

	#global_step = tf.Variable(0, trainable=False, name='global_step')
	batch_size = 2

	## Read data
	testset = set_data_list(args.input, args.data)
	te_total_batch = int(len(testset) / batch_size)

	X = tf.placeholder(tf.float32, [batch_size, 1980, 11])
	output = SNPPro(X, batch_size, training=False)

	## Opertion part
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	sess = tf.Session(config=config)
	saver = tf.train.Saver(max_to_keep=10000) 
	ckpt = tf.train.get_checkpoint_state(ckpt_dir)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		saver = tf.train.import_meta_graph(
			os.path.join(ckpt_dir, 'param-30.meta')
		)
		saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
		print("RESTORE")
	else:
		raise RuntimeError('Does not exist trained parameters.\n')

	o_prediction = os.path.join(args.version, 'prediction_result.csv')
	PRED = open(o_prediction, 'w')
	for inputs, _, genes in get_batch_data(testset, batch_size):
		prediction_output = sess.run(output, feed_dict={X: inputs})
		for i in range(batch_size):
			PRED.write(f'{genes[i]},{prediction_output[i][0]}\n')

	PRED.close()
