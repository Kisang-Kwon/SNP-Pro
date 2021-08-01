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
	parser.add_argument('-t', '--trainset', required=True)
	parser.add_argument('-a', '--validset', required=True)
	parser.add_argument('-v', '--version', required=True, type=str)
	parser.add_argument('-d', '--data', required=True)
	parser.add_argument('-r', '--restore', action='store_true')
	
	return parser.parse_args()


if __name__ == '__main__':
	args = get_arguments()

	o_train = os.path.join(args.version, 'train_result.txt')
	tensorboard = os.path.join(args.version, 'tb')
	ckpt_dir = os.path.join(args.version, 'ckpt')
	dircheck(ckpt_dir)

	global_step = tf.Variable(0, trainable=False, name='global_step')
	batch_size = 2
	dropout_rate = 0.3
	learning_rate = 0.0001
	Epochs = 30

	#batch_size = args.batch_size
	#dropout_rate = args.dropout_rate
	#learning_rate = args.learning_rate
	#Epochs = args.epoch

	## Read data
	trainset = set_data_list(args.trainset, args.data)
	validset = set_data_list(args.validset, args.data)
	train_total_batch = int(len(trainset) / batch_size)
	val_total_batch = int(len(validset) / batch_size)

	X = tf.placeholder(tf.float32, [batch_size, 1980, 11])
	Y = tf.placeholder(tf.float32, [batch_size])
	output = SNPPro(X, batch_size, dropout_rate, training=True)
	loss = tf.sqrt(tf.reduce_mean(tf.square(output - Y)))
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

	total_cost = tf.placeholder(tf.float32)
	tf.summary.scalar('total_cost', total_cost)

	## Opertion part
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	#config.allow_soft_placement = True

	sess = tf.Session(config=config)
	saver = tf.train.Saver(max_to_keep=10000) 
	ckpt = tf.train.get_checkpoint_state(ckpt_dir)
	if args.restore and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		saver = tf.train.import_meta_graph(os.path.join(ckpt_dir, 'param-30.meta'))
		saver.restore(sess, tf.train.latest_checkpoint(ckpt))
		print("RESTORE")
	else:
		sess.run(tf.global_variables_initializer())
		print("RESTART")

	### Training ###
	merged = tf.summary.merge_all()
	writer_train = tf.summary.FileWriter(
		os.path.join(tensorboard, 'train'), sess.graph
	)
	writer_val = tf.summary.FileWriter(
		os.path.join(tensorboard, 'val'), sess.graph
	)

	TRAIN = open(o_train, 'w')
	print('['+time.strftime('%c', time.localtime(time.time()))+'] Training start')
	print(batch_size)
	for epoch in range(1, Epochs+1):
		train_total_cost = 0
		for inputs, labels, genes in get_batch_data(trainset, batch_size):
			_, rmse = sess.run([optimizer, loss], feed_dict={X: inputs, Y: labels})
			train_total_cost += rmse

		avg_train_cost = train_total_cost / train_total_batch
		save_global_step = sess.run(global_step)
		summary_train = sess.run(merged, feed_dict={total_cost: avg_train_cost})
		writer_train.add_summary(summary_train, global_step = epoch)

		val_total_cost = 0
		for inputs, labels, genes in get_batch_data(validset, batch_size):
			rmse = sess.run(loss, feed_dict={X: inputs, Y: labels})
			val_total_cost += rmse

		avg_val_cost = val_total_cost / val_total_batch
		summary_val = sess.run(merged, feed_dict={total_cost: avg_val_cost})
		writer_val.add_summary(summary_val, global_step=epoch)

		print(f'Epoch: {epoch}')
		print(f'Training Avg.cost = {avg_train_cost}')
		print(f'Validation Avg.cost = {avg_val_cost}')
		
		TRAIN.write(f'Epoch: {epoch}\n')
		TRAIN.write(f'Training Avg.cost = {avg_train_cost}\n')
		TRAIN.write(f'Validation Avg.cost = {avg_val_cost}\n')
		
		if epoch % 5 == 0:
			saver.save(sess, os.path.join(ckpt_dir, 'param'), global_step=epoch)
	
	TRAIN.close()
	print('['+time.strftime('%c', time.localtime(time.time()))+'] Complete Optimization')