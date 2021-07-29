import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import time

from utils import read_input_data, input_prep
from dircheck import dircheck
from model import SNPPro
from metrics import correlation, rmse, scatter_plot

## File input
def get_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', nargs=2)
	parser.add_argument('-v', '--version', required=True, type=str)
	parser.add_argument('-ckpt', '--checkpoint', type=str)
	parser.add_argument('-r', '--restore', action='store_true')
	parser.add_argument('-b', '--batch_size', type=int)
	parser.add_argument('-d', '--dropout_rate', type=float)
	parser.add_argument('-l', '--learning_rate', type=float)
	parser.add_argument('-e', '--epoch', type=int, default=30)
	
	return parser.parse_args()


if __name__ == '__main__':
	args = get_arguments()

	ckpt_dir = os.path.join(args.version, 'ckpt')
	dircheck(ckpt_dir)

	global_step = tf.Variable(0, trainable=False, name='global_step')
	batch_size = args.batch_size
	dropout_rate = args.dropout_rate
	learning_rate = args.learning_rate
	Epochs = args.epoch

	## Read data
	te_inputs, te_genes, te_labels = read_input_data(args.input[0], args.input[1])
	te_total_batch = int(len(te_inputs) / batch_size)

	X = tf.placeholder(tf.float32, [batch_size, 1980, 11])
	output = SNPPro(X, batch_size, dropout_rate, training=True)

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
	predictions = []
	labels = []
	genes = []
	for inputs, label, gene in input_prep(batch_size, te_inputs, te_labels, te_genes):
		prediction_output = sess.run(output, feed_dict={X: inputs})
		predictions.extend(prediction_output[0])
		labels.extend(label)
		genes.extend(gene)

		for i in range(batch_size):
			PRED.write(f'{gene[i]},{prediction_output[0][i]}\n')
	
	predictions = np.array(predictions)
	labels = np.array(labels)

	pearson, p_pearson, spearman, p_spearman = correlation(predictions, labels)
	total_rmse = rmse(predictions, labels)

	f_result = os.path.join(args.version, 'result_summary.txt')
	with open(f_result, 'w') as RESULT:
		RESULT.write(f'Pearson Correlation: {pearson} (p-value: {p_pearson})\n')
		RESULT.write(f'Spearman Correlation: {spearman} (p-value: {p_spearman})\n')
		RESULT.write(f'RMSE: {total_rmse}\n')

	f_scatter = os.path.join(args.version, 'prediction_scatter.png')
	scatter_plot(predictions, labels, genes, f_scatter)
	PRED.close()