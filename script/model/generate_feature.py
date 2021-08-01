import os
import numpy as np

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


def get_gene_name():
    fpath = '../../data/processed_data/bin_number.txt'
    gene_info = dict()
    with open(fpath) as Fopen:
        for line in Fopen:
            if line.startswith('#'):
                pass
            else:
                arr = line.rstrip().split('\t')
                chrom = f'chr{arr[0]}'
                tss = int(arr[1])
                label = arr[3]

                gene_info[label] = f'{chrom}_{tss}'

    return gene_info


if __name__ == '__main__':
    filelist = [
        '../../data/processed_data/train/HG02601/20000/input.20000.txt',
        '../../data/processed_data/train/HG02601/10000/input.10000.txt',
        '../../data/processed_data/train/HG02601/5000/input.5000.txt',
        '../../data/processed_data/train/HG02601/1000/input.1000.txt'
    ]

    gene_info = get_gene_name()

    for fpath in filelist:
        with open(fpath) as Fopen:
            for line in Fopen:
                arr = line.rstrip().split('\t')
                gene = gene_info[arr[0]]
                feature = np.array(onehot_encoding(arr[1]))
                
                out_fpath = f'../../data/features/HG02601/{gene}.npy'
                np.save(
                    out_fpath,
                    np.array([gene, feature], dtype=object)
                )