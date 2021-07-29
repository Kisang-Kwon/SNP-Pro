import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.stats import pearsonr, spearmanr

def correlation(inputs, labels):
    print(inputs.shape, labels.shape)
    r, p_val_pearson = pearsonr(inputs, labels)
    rho, p_val_spearman = spearmanr(inputs, labels)

    return r, p_val_pearson, rho, p_val_spearman

def rmse(inputs, labels):
    return np.sqrt(np.mean(np.square(inputs - labels)))

def scatter_plot(inputs, labels, genes, fpath):
    df = pd.DataFrame(zip(genes, inputs, labels), columns=['Genes', 'Predictions', 'Labels'])

    #plt.figure(figsize=(10,8))
    plt.title('Scatter Plot')
    plt.xlim([0, 52])
    plt.ylim([0, 52])
    plt.xlabel('Predictions')
    plt.ylabel('Promoter Activities')
    
    sns.scatterplot(data=df, x='Predictions', y='Labels')
    #sns.regplot(data=df, x='Predictions', y='Labels', fit_reg=True)

    plt.savefig(fpath, bbox_inches='tight')