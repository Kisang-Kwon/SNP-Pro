import numpy as np


def read_input_data(f_input, f_label=None):
    input_list = []
    label_list = []
    gene_list = []
    with open(f_input) as Fopen:
        for line in Fopen:
            arr = line.rstrip().split('\t')
            gene_list.append(arr[0])
            input_list.append(arr[1])

    if f_label is not None:
        with open(f_label) as Fopen:
            for line in Fopen:
                arr = line.rstrip().split('\t')
                label_list.append(float(arr[1]))

        return input_list, gene_list, label_list
    else:
        return input_list, gene_list

def onehot_encoding(sequence):
	nt_dict = {
		'0':[1,0,0,0,0,0,0,0,0,0,0],
		'1':[0,1,0,0,0,0,0,0,0,0,0],
		'2':[0,0,1,0,0,0,0,0,0,0,0],
		'3':[0,0,0,1,0,0,0,0,0,0,0],
		'4':[0,0,0,0,1,0,0,0,0,0,0],
		'5':[0,0,0,0,0,1,0,0,0,0,0],
		'6':[0,0,0,0,0,0,1,0,0,0,0],
		'7':[0,0,0,0,0,0,0,1,0,0,0],
		'8':[0,0,0,0,0,0,0,0,1,0,0],
		'9':[0,0,0,0,0,0,0,0,0,1,0],
		'N':[0,0,0,0,0,0,0,0,0,0,1]
	}

	tmp_seq = []
	for i in sequence:
		tmp_seq.append(nt_dict.get(i))
	
	return tmp_seq

def input_prep(batch_size, input_list, label_list, gene_list):
    t_input = []
    t_label = []
    t_genes = []
    for data, label, gene in zip(input_list, label_list, gene_list):
        t_input.append(onehot_encoding(data))
        t_label.append(float(label))
        t_genes.append(gene)

        if len(t_label) == batch_size:
            t_input = np.array(t_input)
            t_label = np.array(t_label)
            yield t_input, t_label, t_genes

            t_input = []
            t_label = []
            t_genes = []
    
    #t_input = np.array(t_input)
    #t_label = np.array(t_label)
    #yield t_input, t_label, t_genes

def input_prep_for_predict(batch_size, input_list, gene_list):
    t_input = []
    t_genes = []

    for data, gene in zip(input_list, gene_list):
        t_input.append(onehot_encoding(data))
        t_genes.append(gene)

        if len(t_input) == batch_size:
            t_input = np.array(t_input)
            
            yield t_input, t_genes

            t_input = []
            t_genes = []
    
    #t_input = np.array(t_input)
    #yield t_input, t_genes