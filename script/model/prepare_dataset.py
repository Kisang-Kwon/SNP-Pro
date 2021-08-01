import os
import sys
import math
import csv
import random
import numpy as np
import tensorflow as tf

from multiprocessing import Pool


def set_data_list(f_datalist, data_dir):
    Fopen = open(f_datalist)
    cread = csv.reader(Fopen)
    
    data_list = []
    for line in cread:
        gene = line[0]
        label = float(line[1])

        fname = f'{gene}.npy'
        fpath = os.path.join(data_dir, fname)
        #fname = f'tss_{chrom}_{tss}.npy'
        #fpath = os.path.join(data_dir, sample, 'input', fname)
        data_list.append([fpath, label, gene])
        
    Fopen.close()
    
    random.shuffle(data_list)

    return data_list

def get_batch_data(dataset, batch_size):
    matrices = []
    labels = []
    info = []
    for data in dataset:
        fpath = data[0]
        label = float(data[1])
        gene = data[2]

        if len(matrices) == batch_size:
            matrices = np.array(matrices, dtype='float32')
            labels = np.array(labels, dtype='float32')
            yield matrices, labels, info

            matrices = []
            labels = []
            info = []

        matrix = np.load(fpath, allow_pickle=True)[1]
        matrices.append(matrix)
        labels.append(label)
        info.append(gene)
    
    #matrices = tf.convert_to_tensor(matrices, dtype='float32')
    #matrices = tf.expand_dims(matrices, axis=3)
    #labels = tf.convert_to_tensor(labels, dtype='float32')
    #yield matrices, labels, info