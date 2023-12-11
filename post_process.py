
__author__ = 'Bahram Jafrasteh'
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
P=1
Ss = [1, 4, 10, 15][::-1]  # Repeat the K-fold cross-validation S times
Ks = [2, 10, 50, 100, 200, 400]  # K-fold CV
from itertools import product
dataset='abcd5'
Es = [5]  # Level of perturbation
Ns = [1000]  # Number of samples

def create_box_plots(data, labels, title='BoxPlots'):

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.boxplot(data, labels=labels,notch=True, patch_artist=True)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    plt.tight_layout()

    ax.set_title(title)

    # Show the plot
    plt.savefig('boxplot_{}_{}.pdf'.format(dataset, title))
    plt.close()

def make_dic(comb):
    km = defaultdict(list)  # np.zeros((len(Ss), len(Ks)))
    for el in comb:
        km[el] = []
    return km
def calc_mean(dicti, shape):
    m = np.zeros(shape)
    for key in dicti.keys():
        m[key[0], key[1]]=np.nanmean(dicti[key])
    return m

def calc_median(dicti, shape):
    m = np.zeros(shape)
    for key in dicti.keys():
        arr = np.array(dicti[key])
        non_nan_values = arr[~np.isnan(arr)]
        m[key[0], key[1]]=np.median(non_nan_values)
    return m


with open('{}_perturbation_{}_sample_{}.pkl'.format(dataset, Es[0], Ns[0]), 'rb') as file:
    dictionary = pickle.load(file)

comb = product([i for i in range(len(Ss))], [i for i in range(len(Ks))])


KM_mcnemar = make_dic(comb)
KM_delong = make_dic(comb)#np.zeros((len(Ss), len(Ks)))
KM_t_test = make_dic(comb)#np.zeros((len(Ss), len(Ks)))
KM_corrected_t_test = make_dic(comb)#np.zeros((len(Ss), len(Ks)))

# add the results to a the matrix
for i, el in enumerate(dictionary.keys()):
    combinations = dictionary[el][0]
    results = dictionary[el][1]

    [Pv_mcnemar, Pv_delong, p_value_t_test, p_value_corrected_t_test] = results
    row, col = Ss.index(combinations[2]), Ks.index(combinations[0])
    KM_mcnemar[row, col].append([Pv_mcnemar])
    KM_delong[row, col].append( Pv_delong)
    KM_t_test[row, col].append([ p_value_t_test])
    KM_corrected_t_test[row, col].append([ p_value_corrected_t_test])

shape = (len(Ss), len(Ks))
### MEDIAN ####
KM_mcnemar_median = calc_median(KM_mcnemar, shape)
KM_delong_median = calc_median(KM_delong, shape)
KM_t_test_median = calc_median(KM_t_test, shape)
KM_corrected_t_test_median = calc_median(KM_corrected_t_test, shape)
create_box_plots([KM_t_test_median.ravel(), KM_corrected_t_test_median.ravel(), KM_mcnemar_median.ravel(), KM_delong_median[~np.isnan(KM_delong_median)].ravel()],
                 labels=['Uncorrected t-test', 'Corrected t-test', 'McNemar', 'Delong'], title='Median')
######### MEAN###
KM_mcnemar = calc_mean(KM_mcnemar, shape)
KM_delong = calc_mean(KM_delong, shape)
KM_t_test = calc_mean(KM_t_test, shape)
KM_corrected_t_test = calc_mean(KM_corrected_t_test, shape)
create_box_plots([KM_t_test.ravel(), KM_corrected_t_test.ravel(), KM_mcnemar.ravel(), KM_delong[~np.isnan(KM_delong)].ravel()],
                 labels=['Uncorrected t-test', 'Corrected t-test', 'McNemar', 'Delong'], title='Mean')
import seaborn as sns


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 6))

# Set up the Seaborn heatmap
sns.set()


x_labels = [str(i) for i in Ks]
y_labels = [str(i) for i in Ss]
titles = ['Corrected t-test', 't-test', 'Corrected t-test (median)', 't-test (median)']
Matrices = [KM_corrected_t_test, KM_t_test, KM_corrected_t_test_median, KM_t_test_median]

for ii, (i,j) in enumerate(product([0,1], repeat=2)):

    sns.heatmap(Matrices[ii], cmap="viridis", annot=True, fmt=".2f", linewidths=.5, ax=axes[i,j])# vmin=0, vmax=1)
    axes[i,j].set_title(titles[ii])
    axes[i,j].set_xticks(np.arange(0.5, len(x_labels)+.5))
    axes[i,j].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[i,j].set_yticks(np.arange(0.5, len(y_labels)+.5))
    axes[i,j].set_yticklabels(y_labels, rotation=45, ha='right')

#cbar = plt.colorbar(Matrices[ii], ax=axes, orientation='vertical', pad=0.04)

plt.tight_layout()

plt.savefig('Plots_{}.pdf'.format(dataset))
plt.close()
