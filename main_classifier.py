__author__ = 'Bahram Jafrasteh'
# based on matlab code written by Qingyu Zhao

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
import compare_auc_delong_xu


dataset='abcd5'
import pandas as pd

if dataset.lower()=='abcd5':


    df_thk = pd.read_csv('abcd-data-release-5.0/core/imaging/mri_y_smr_thk_dsk.csv')
    df_area = pd.read_csv('abcd-data-release-5.0/core/imaging/mri_y_smr_area_dsk.csv')
    df_aseg = pd.read_csv('abcd-data-release-5.0/core/imaging/mri_y_smr_vol_aseg.csv')
    df_demo = pd.read_csv('abcd-data-release-5.0/core/abcd-general/abcd_p_demo.csv')

    sub_df_thk = df_thk[df_thk['eventname']=='baseline_year_1_arm_1']
    sub_df_area = df_area[df_area['eventname']=='baseline_year_1_arm_1']
    sub_df_aseg = df_aseg[df_aseg['eventname']=='baseline_year_1_arm_1']
    sub_df_demo = df_demo[df_demo['eventname']=='baseline_year_1_arm_1']

    'demo_sex_v2'
    #merged_df = pd.merge(df_demo, df_thk, on='src_subject_id')
    #merged_df[[el for el in list(df_thk.columns) + ['demo_sex_v2'] if el != 'eventname']]
    subject_id = sub_df_thk['src_subject_id'].values
    sub_df_thk = sub_df_thk.iloc[:, 2:70]

    av_left_right_df_thk = (sub_df_thk.iloc[:, :34].values + sub_df_thk.iloc[:, 34:].values)/2

    sub_df_area = sub_df_area.iloc[:, 2:70]
    av_left_right_df_area = (sub_df_area.iloc[:, :34].values + sub_df_area.iloc[:, 34:].values)/2

    sub_df_aseg_l = sub_df_aseg.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 19]]
    sub_df_aseg_r = sub_df_aseg.iloc[:, [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33]]
    av_left_right_df_aseg = (sub_df_aseg_l.values + sub_df_aseg_r.values)/2

    rest_aseg = sub_df_aseg.iloc[:, [12, 13, 16, 18, 32, 34, 37, 38, 39, 40, 41, 42]].values

    all_features = np.concatenate([av_left_right_df_thk, av_left_right_df_area, av_left_right_df_aseg, rest_aseg], 1)

    df_features = pd.DataFrame(all_features)
    col_features = ['feature_{}'.format(el+1) for el in range(all_features.shape[1])]
    df_features.columns = col_features
    df_features['src_subject_id'] = subject_id
    merged_df = pd.merge(df_demo, df_features, on='src_subject_id')

    df_features_output = merged_df[col_features + ['demo_sex_v2']].dropna(axis=0)
    features_not_processed = df_features_output[col_features].values

    # regressing out head size
    headsize = sub_df_aseg['smri_vol_scs_suprateialv'].values
    model = LinearRegression(fit_intercept=True)
    for i in range(features_not_processed.shape[1]):
        model.fit(headsize.reshape(-1,1), features_not_processed[:,i])
        features_not_processed[:,i] -= model.coef_*headsize

    labels = df_features_output['demo_sex_v2'].values-1
    ind_lable = labels<2
    features_not_processed = features_not_processed[ind_lable,:]
    labels = labels[ind_lable]
    scaler = StandardScaler()
    inputData = scaler.fit_transform(features_not_processed)

def McNemar(y_predp, y_predn, y_test):
    # Create a contingency table

    table = [[sum((y_predp == y_test) & (y_predn == y_test)),
                          sum((y_predp == y_test) & (y_predn != y_test))],
                         [sum((y_predp != y_test) & (y_predn == y_test)),
                          sum((y_predp != y_test) & (y_predn != y_test))]]
    # Perform McNemar's test
    from statsmodels.stats.contingency_tables import mcnemar
    results = mcnemar(table)
    return results.pvalue

def DeLong(y_predp, y_predn, y_test):
    # Perform DeLong's test
    log_p = compare_auc_delong_xu.delong_roc_test(
        y_test.squeeze().astype('int'), y_predn.squeeze(), y_predp.squeeze())
    return np.exp(log_p).item()

def t_test(accp , accn):
    from scipy.stats import ttest_rel
    _, p_value = ttest_rel(accp, accn)
    return p_value



def corrected_t_test(accp , accn, N, K):
    """
    perform corrected t-test
    :param accp: accuracy 1
    :param accn: accuracy 2
    :param N: total number of dataset
    :param K: number of folds
    :return:
    """
    from scipy.stats import t

    accd = accp - accn
    accm = np.mean(accd)
    n = len(accd)
    n1 = N-N/K; n2 = N/K
    sigma2 = sum((accd-accm)**2)/(n-1)
    sigma2_mod = sigma2 * (1/n + n2/n1)
    t_static =  accm / np.sqrt(sigma2_mod)
    return t.cdf(-abs(t_static),n-1)

def cross_validation(inputData, labels, kf, perturb):
    """
    Performing K-fold cross validaiton
    :param inputData: X
    :param labels: T
    :param kf: kfold
    :param perturb: perturbation matrix
    :return: accuracies and last prediction
    """
    accpos = []
    accneg = []
    for train_index, test_index in kf.split(inputData, labels):
        X_train, X_test = inputData[train_index], inputData[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        model = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)
        model.fit(X_train, y_train.squeeze())
        coef_ = model.coef_.copy()
        #model2 = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)
        #model2.fit(X_train, y_train.squeeze())

        coefp = coef_+perturb
        coefn = coef_-perturb
        model.coef_ = coefp
        y_predp = model.predict(X_test)
        model.coef_ = coefn
        y_predn = model.predict(X_test)
        accp = accuracy_score(y_predp, y_test)
        accn = accuracy_score(y_predn, y_test)
        accpos.append(accp)
        accneg.append(accn)
    return accpos, accneg, [y_predp, y_predn, y_test]

def select_sample(ind_gender, size):
    inp1 = inputData[ind_gender, :]
    lb1 = labels[ind_gender]
    randomindinces_p = np.random.choice((ind_gender).sum(),size )
    X_sel = inp1[randomindinces_p, :]
    Y_sel = lb1[randomindinces_p]
    return X_sel, Y_sel
def run_experiment(info):
    """
    Run the experiment
    :param info:
    :return:
    """
    K, E, S, N =info
    print(info)
    # K number of folds in cross-validation
    # E perturbation level
    # S number of repetition of the K-fold cross-validation
    ind_gender = (labels==1)
    x1,y1=select_sample(ind_gender, N//2)
    x2, y2=select_sample(~ind_gender, N-N//2)

    X_sel = np.concatenate([x1,x2])
    Y_sel = np.concatenate([y1,y2])
    feature = X_sel.shape[1] #number of features
    perturb = np.random.randn(1,feature) / E # perturbation matrix

    all_accps = []
    all_accneg = []
    Pv_mcnemar = []
    Pv_delong = []
    for i in range(S):
        # stratified k-fold cross validation
        kf = StratifiedKFold(n_splits=K, shuffle=True)
        accpos, accneg, [y_predp, y_predn, y_test] = cross_validation(X_sel, Y_sel, kf, perturb)
        all_accps.append(np.array(accpos))
        all_accneg.append(np.array(accneg))

        # perfrom McNemar test
        pv_mcnemar = McNemar(y_predp, y_predn, y_test)
        Pv_mcnemar.append(pv_mcnemar)
        # Perform DeLong's test
        try:
            pvalue_delong = DeLong(y_predp, y_predn, y_test)
            Pv_delong.append(pvalue_delong)
        except:
            pass


    all_accps = np.stack(all_accps).ravel()
    all_accneg = np.stack(all_accneg).ravel()


    # perform t_test
    p_value_t_test = t_test(all_accps, all_accneg)

    if np.isnan(p_value_t_test):
        print('')
    # perform corrected t_test
    p_value_corrected_t_test = corrected_t_test(all_accps, all_accneg, N, K)

    return [np.nanmean(Pv_mcnemar), np.nanmean(Pv_delong), p_value_t_test, p_value_corrected_t_test]





if __name__ == '__main__':
    from collections import defaultdict
    import pickle

    P = 100 # repeat the whole experiment P times
    Ss = [1, 4, 10, 15] # Repeat the K-fold cross-validation S times
    Ks = [2, 10, 50, 100, 200, 400] # K-fold CV
    #Ss=[1]
    #Ks=[100]
    Es = [5] # Level of perturbation
    Ns = [1000] # Number of samples

    list_total = []
    for p in range(P):
        for k in Ks:
            for s in Ss:
                for e in Es:
                    for n in Ns:
                        list_total.append([k, e, s, n])


    #for el in list_total:
    #    run_experiment(el)
    pool = mp.Pool(int(mp.cpu_count() // 10))
    results = pool.map(run_experiment, list_total)
    dictionary = defaultdict(list)
    for i, el in enumerate(list_total):
        dictionary[i] = [el, results[i]]
    # write the results to a file
    with open('{}_perturbation_{}_sample_{}.pkl'.format(dataset, Es[0], Ns[0]), 'wb') as file:
        pickle.dump(dictionary, file)
