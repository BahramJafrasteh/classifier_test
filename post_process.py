
__author__ = 'Bahram Jafrasteh'
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import pandas as pd



def create_box_plots(data, labels, title='BoxPlots', ax=None):

    ax.boxplot(data, labels=labels,notch=True, patch_artist=True)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    plt.tight_layout()

    #ax.set_title(title)
    ax.set_ylabel(title)
    # Show the plot
    #plt.savefig('boxplot_{}_{}.pdf'.format(dataset, title))

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




def calc_numk(dicti, shape, p):
    """
    Calculate P-values for each k (Kfold)
    :param dicti:
    :param shape:
    :param p:
    :return:
    """
    dd = defaultdict()
    for i in range(shape[1]):
        dd[i] = []
    for key in dicti.keys():
        #print(key[1])
        arr = np.array(dicti[key])
        non_nan_values = arr[~np.isnan(arr)]
        dd[key[1]].append(non_nan_values)
    m = np.zeros((shape[0]*p, shape[1]))
    for key in dd.keys():
        a = np.stack(dd[key]).ravel()
        m[:a.shape[0], key] = a
        m[a.shape[0]:, key] = np.nan
    return m



def generate_boxplots_mean_median_pr(KM_mcnemar, KM_delong, KM_t_test, KM_corrected_t_test):

    ######### MEAN###
    KM_mcnemar_mean = calc_mean(KM_mcnemar, shape)
    KM_delong_mean = calc_mean(KM_delong, shape)
    KM_t_test_mean = calc_mean(KM_t_test, shape)

    ### MEDIAN ####
    KM_mcnemar_median = calc_median(KM_mcnemar, shape)
    KM_delong_median = calc_median(KM_delong, shape)
    KM_t_test_median = calc_median(KM_t_test, shape)
    KM_corrected_t_test_median = calc_median(KM_corrected_t_test, shape)

    ### PR ####
    KM_mcnemar_pr = calc_pr(KM_mcnemar, shape)
    KM_delong_pr = calc_pr(KM_delong, shape)
    KM_t_test_pr = calc_pr(KM_t_test, shape)
    KM_corrected_t_test_pr = calc_pr(KM_corrected_t_test, shape)

    # Create a figure and axis
    fig, ax = plt.subplots(3, 1, figsize=(5, 10))
    create_box_plots([KM_t_test_median.ravel(), KM_corrected_t_test_median.ravel(), KM_mcnemar_median.ravel(),
                      KM_delong_median[~np.isnan(KM_delong_median)].ravel()],
                     labels=['Uncorrected t-test', 'Corrected t-test', 'McNemar', 'Delong'], title='Median', ax=ax[0])

    KM_corrected_t_test_mean = calc_mean(KM_corrected_t_test, shape)
    create_box_plots([KM_t_test_mean.ravel(), KM_corrected_t_test_mean.ravel(), KM_mcnemar_mean.ravel(),
                      KM_delong_mean[~np.isnan(KM_delong_mean)].ravel()],
                     labels=['Uncorrected t-test', 'Corrected t-test', 'McNemar', 'Delong'], title='Mean', ax=ax[1])

    create_box_plots([KM_t_test_pr.ravel(), KM_corrected_t_test_pr.ravel(), KM_mcnemar_pr.ravel(),
                      KM_delong_pr[~np.isnan(KM_delong_pr)].ravel()],
                     labels=['Uncorrected t-test', 'Corrected t-test', 'McNemar', 'Delong'], title='PR', ax=ax[2])

    plt.show()
    plt.close()
    return [KM_t_test_mean, KM_corrected_t_test_mean, KM_t_test_median, KM_corrected_t_test_median,
 KM_t_test_pr, KM_corrected_t_test_pr]


def generate_KM_box(KM_t_test_mean, KM_corrected_t_test_mean, KM_t_test_median, KM_corrected_t_test_median,
                KM_t_test_pr, KM_corrected_t_test_pr):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(5, 6))

    # Set up the Seaborn heatmap
    sns.set()

    x_labels = [str(i) for i in Ks]
    y_labels = [str(i) for i in Ss]
    titles = ['t-test', 'Corrected t-test', 't-test (median)', 'Corrected t-test (median)',
              't-test (pr)', 'Corrected t-test (pr)']
    Matrices = [KM_t_test_mean, KM_corrected_t_test_mean, KM_t_test_median, KM_corrected_t_test_median,
                KM_t_test_pr, KM_corrected_t_test_pr]

    for ii, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]):

        sns.heatmap(Matrices[ii], cmap="viridis", annot=False, fmt=".2f", linewidths=.5,
                    ax=axes[i, j])  # vmin=0, vmax=1)
        axes[i, j].set_title(titles[ii])
        axes[i, j].set_xticks(np.arange(0.5, len(x_labels) + .5))
        axes[i, j].set_xticklabels(x_labels, rotation=45, ha='right')
        axes[i, j].set_yticks(np.arange(0.5, len(y_labels) + .5))
        axes[i, j].set_yticklabels(y_labels, rotation=45, ha='right')
        if 'median' in titles[ii]:
            axes[i, j].set_ylabel('median')
        elif 'pr' in titles[ii]:
            axes[i, j].set_ylabel('pr')
        else:
            axes[i, j].set_ylabel('mean')
    # cbar = plt.colorbar(Matrices[ii], ax=axes, orientation='vertical', pad=0.04)

    plt.tight_layout()
    plt.show()
    # plt.savefig('Plots_{}.pdf'.format(dataset))
    plt.close()

def generate_kfold_k(KM_delong, KM_mcnemar, shape, P):
    """
    Generate group box plots
    :param KM_delong:
    :param KM_mcnemar:
    :param shape:
    :param P:
    :return:
    """
    KM_delong_k = calc_numk(KM_delong, shape, P)
    KM_mcnemar_k = calc_numk(KM_mcnemar, shape, P)

    mm = np.zeros((np.prod(KM_delong_k.shape) * 2, 3))
    r = 0
    aug = KM_delong_k.shape[0]
    for i, k in enumerate(Ks):
        mm[r:r + aug, 0] = KM_delong_k[:, i]
        mm[r:r + aug, 1] = k
        mm[r:r + aug, 2] = 0
        r += aug
    for i, k in enumerate(Ks):
        mm[r:r + aug, 0] = KM_mcnemar_k[:, i]
        mm[r:r + aug, 1] = k
        mm[r:r + aug, 2] = 1
        r += aug
    df_data = pd.DataFrame(mm)
    df_data.columns = ['P_values', 'K-fold', 'Method']
    df_data.loc[df_data['Method'] == 0, 'Method'] = 'DeLong'
    df_data.loc[df_data['Method'] == 1, 'Method'] = 'McNemar'
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.boxplot(x=df_data['K-fold'],
                y=df_data['P_values'],
                hue=df_data['Method'],
                palette='Set2', ax=ax)
    plt.show()


def get_matrix_data(dictionary):
    comb = product([i for i in range(len(Ss))], [i for i in range(len(Ks))])
    KM_mcnemar = make_dic(comb)
    KM_delong = make_dic(comb)  # np.zeros((len(Ss), len(Ks)))
    KM_t_test = make_dic(comb)  # np.zeros((len(Ss), len(Ks)))
    KM_corrected_t_test = make_dic(comb)  # np.zeros((len(Ss), len(Ks)))

    # add the results to a the matrix
    for i, el in enumerate(dictionary.keys()):
        combinations = dictionary[el][0]
        results = dictionary[el][1]

        [Pv_mcnemar, Pv_delong, p_value_t_test, p_value_corrected_t_test] = results
        row, col = Ss.index(combinations[2]), Ks.index(combinations[0])
        KM_mcnemar[row, col].append([Pv_mcnemar])
        KM_delong[row, col].append(Pv_delong)
        KM_t_test[row, col].append([p_value_t_test])
        KM_corrected_t_test[row, col].append([p_value_corrected_t_test])
    return KM_mcnemar, KM_delong, KM_t_test, KM_corrected_t_test

def calc_pr(dicti, shape):
    m = np.zeros(shape)
    for key in dicti.keys():
        arr = np.array(dicti[key])
        non_nan_values = arr[~np.isnan(arr)]
        m[key[0], key[1]]=(non_nan_values<0.01).sum()/non_nan_values.shape[0]
    return m


if __name__=='__main__':
    global P, Ss, Ks, dataset, Es, Ns
    P = 100
    Ss = [1, 4, 10, 15][::-1]  # Repeat the K-fold cross-validation S times
    Ks = [2, 10, 50, 100, 200, 400]  # K-fold CV
    from itertools import product

    dataset = 'abcd5'
    dataset = 'adni'
    Es = [5]  # Level of perturbation
    Ns = [1000]  # Number of samples

    print('{}_perturbation_{}_sample_{}.pkl'.format(dataset, Es[0], Ns[0]))
    with open('{}_perturbation_{}_sample_{}.pkl'.format(dataset, Es[0], Ns[0]), 'rb') as file:
        dictionary = pickle.load(file)

    KM_mcnemar, KM_delong, KM_t_test, KM_corrected_t_test = get_matrix_data(dictionary)
    shape = (len(Ss), len(Ks))


    ### Generate grouped boxplots ####
    generate_kfold_k(KM_delong, KM_mcnemar, shape, P)

    ### Generate three boxplots ####
    [KM_t_test_mean, KM_corrected_t_test_mean, KM_t_test_median, KM_corrected_t_test_median,
     KM_t_test_pr, KM_corrected_t_test_pr] = generate_boxplots_mean_median_pr(KM_mcnemar, KM_delong, KM_t_test, KM_corrected_t_test)

    generate_KM_box(KM_t_test_mean, KM_corrected_t_test_mean, KM_t_test_median, KM_corrected_t_test_median,
                    KM_t_test_pr, KM_corrected_t_test_pr)
