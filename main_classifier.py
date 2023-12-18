__author__ = 'Bahram Jafrasteh'
# based on matlab code written by Qingyu Zhao
import torch
import torch.nn as nn
import os
import h5py
import numpy as np
import multiprocessing as mp
from collections import defaultdict
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
import compare_auc_delong_xu
import sys
from torchvision import transforms
from MLP_model import MLPClassifier, train_perturb_evaluate_MLP_model, calc_accuracy



import pandas as pd
def get_data(dataset_name):
    if dataset_name.lower()=='abcd5':


        df_thk = pd.read_csv('abcd-data-release-5.0/core/imaging/mri_y_smr_thk_dsk.csv')
        df_area = pd.read_csv('abcd-data-release-5.0/core/imaging/mri_y_smr_area_dsk.csv')
        df_aseg = pd.read_csv('abcd-data-release-5.0/core/imaging/mri_y_smr_vol_aseg.csv')
        df_demo = pd.read_csv('abcd-data-release-5.0/core/abcd-general/abcd_p_demo.csv')

        sub_df_thk = df_thk[df_thk['eventname']=='baseline_year_1_arm_1']
        sub_df_area = df_area[df_area['eventname']=='baseline_year_1_arm_1']
        sub_df_aseg = df_aseg[df_aseg['eventname']=='baseline_year_1_arm_1']
        sub_df_demo = df_demo[df_demo['eventname']=='baseline_year_1_arm_1']

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
        inputData = features_not_processed[ind_lable,:]
        labels = labels[ind_lable]

    elif dataset_name.lower()=='adni':

        df_head = pd.read_csv('ADNI/ADNIMERGE.csv')
        df_head = df_head.dropna(subset=['DX']) # drop all na values from diagnostics
        df_head = df_head[df_head['VISCODE'] == 'bl']
        df_1 = pd.read_csv('ADNI/roi_data_ADNI/roi/UCSFFSX_11_02_15_28Sep2023.csv')
        df_2 = pd.read_csv('ADNI/roi_data_ADNI/roi/UCSFFSX6_08_17_22_28Sep2023.csv')
        df_3 = pd.read_csv('ADNI/roi_data_ADNI/roi/UCSFFSX51_11_08_19_28Sep2023.csv')
        df_4 = pd.read_csv('ADNI/roi_data_ADNI/roi/UCSFFSX51_ADNI1_3T_02_01_16_28Sep2023.csv')
        df_con = pd.concat([df_1, df_2, df_3, df_4])

        df_total = pd.merge(df_head, df_con, on='IMAGEUID')
        df_total = df_total[df_total['OVERALLQC'].astype(str) == 'Pass']

        feature_names = ['ST91TA', 'ST32TA', 'ST13CV', 'ST115CV', 'ST26CV', 'ST118CV', 'ST13TA', 'ST26TA', 'ST29SV', 'ST72CV', 'ST103CV',
         'ST58CV', 'ST91CV', 'ST31CV', 'ST60TA', 'ST111CV', 'ST119TA', 'ST30SV', 'ST129CV', 'ST44TA', 'ST111TA',
         'ST32CV', 'ST24CV', 'ST85TA', 'ST99TA', 'ST99CV', 'ST58TA', 'ST24TA', 'ST50CV', 'ST59CV', 'ST12SV', 'ST130CV',
         'ST40CV', 'ST44CV', 'ST52CV', 'ST109CV', 'ST72TA', 'ST31TA', 'ST83CV', 'ST90TA', 'ST90CV', 'ST89SV', 'ST83TA',
         'ST85CV', 'ST88SV', 'ST52TA', 'ST71SV', 'ST40TA', 'ST117TA', 'ST103TA', 'ST117CV', 'AGE', 'PTGENDER', 'DX_bl', 'ICV']


        df_total = df_total[feature_names]
        mapping = {'Male': 1, 'Female': 0}
        df_total['PTGENDER'] = df_total['PTGENDER'].map(mapping)
        headsize = df_total['ICV'].values
        features_not_processed = df_total[feature_names[:-2]].values.astype(np.float)
        model = LinearRegression(fit_intercept=True)
        for i in [-1,-2]:
            model.fit(headsize.reshape(-1,1), features_not_processed[:,i])
            features_not_processed[:,i] -= model.coef_*headsize
        labels = df_total['DX_bl'].map({'AD':1}).fillna(0).values
        inputData = features_not_processed
    return inputData, labels

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

def cross_validation(Xx, Yy, kf, perturb, info_model, use_model=None, early_break=False, dist_p=None, repeat_no=0):
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
    accs = []
    #r = 0
    [train_indices, val_indices] = kf
    if use_model=='mlp':
        hidden_size, learning_rate, num_epochs, batch_size = info_model
        device = torch.device('cpu')
        if device_id>=0:
            if torch.cuda.is_available():
                #device = torch.device('cuda:{}'.format(device_id))
                if 2>1:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    optimal = None
                    for i, gpu in enumerate(gpus):
                        used = gpu.memoryUsed/gpu.memoryTotal
                        if used<0.9:
                            optimal = i
                            break
                    if optimal is not None:
                        device = torch.device('cuda:{}'.format(optimal))

        Xx = torch.from_numpy(Xx.astype(np.float32)).to(device)
        Yy = torch.from_numpy(Yy.astype(np.float32)).to(device)
        perturb = torch.from_numpy(perturb.astype(np.float32)).to(device)

        input_size = Xx.shape[1]
        output_size = 1
        model = MLPClassifier(input_size, hidden_size, output_size)
        model.to(device)
        state_dict = model.state_dict()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for r, [train_index, test_index] in enumerate(zip(*[train_indices, val_indices])):
        #if early_break and r ==1:
        #    break
        #r+=1
        X_train, X_test = Xx[train_index, :], Xx[test_index, :]
        y_train, y_test = Yy[train_index], Yy[test_index]

        class_weights = y_train.shape[0] / (2 * torch.bincount(y_train.to(torch.int))[1])
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

        if use_model=='lr':


            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)
            model.fit(X_train, y_train.squeeze())

            y_pred = model.predict(X_test)
            coef_ = model.coef_.copy()
            #model2 = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)
            #model2.fit(X_train, y_train.squeeze())

            coefp = coef_+perturb
            coefn = coef_-perturb
            model.coef_ = coefp
            y_predp = model.predict(X_test)
            model.coef_ = coefn
            y_predn = model.predict(X_test)

            acc = accuracy_score(y_pred, y_test)
            #print(acc)
            accp = accuracy_score(y_predp, y_test)
            accn = accuracy_score(y_predn, y_test)

        elif use_model=='mlp':
            # Calculate mean and standard deviation along the specified axis (usually axis=0)
            mean = X_train.mean(0)
            std = X_train.std(0)+1e-10
            # normalize data
            X_train = (X_train-mean)/(std)
            X_test = (X_test-mean)/(std)
            model.load_state_dict(state_dict) #reset model weights
            out_f_name = os.path.join(dist_p,'{}_cv_{:03d}_rep_{}.pth'.format(use_model, r+1, repeat_no))
            y_pred, y_predp, y_predn = train_perturb_evaluate_MLP_model(X_train,y_train, perturb, X_test, model, optimizer, criterion,
                           num_epochs=num_epochs,    batch_size = batch_size, device=device, out_f_name=out_f_name)

            acc = calc_accuracy(y_pred, y_test)
            accp = calc_accuracy(y_predp, y_test)
            accn = calc_accuracy(y_predn, y_test)
        accpos.append(accp)
        accneg.append(accn)
        accs.append(acc)
    if use_model=='mlp':
        y_predn = y_predn.detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()
        y_predp = y_predp.detach().cpu().numpy()
    return accpos, accneg, [y_predp, y_predn, y_test]

def select_sample(ind_gender, size):
    inp1 = inputData[ind_gender, :]
    lb1 = labels[ind_gender]
    if size< lb1.shape[0]:
        randomindinces_p = np.random.choice((ind_gender).sum(),size )
        X_sel = inp1[randomindinces_p, :]
        Y_sel = lb1[randomindinces_p]
    else:
        X_sel = inp1
        Y_sel = lb1

    return X_sel, Y_sel
def run_experiment(info):
    """
    Run the experiment
    :param info:
    :return:
    """
    K, E, S, N,p =info
    print(info)
    # K number of folds in cross-validation
    # E perturbation level
    # S number of repetition of the K-fold cross-validation



    ind_gender = (labels==1)
    x1,y1=select_sample(ind_gender, N//2)
    x2, y2=select_sample(~ind_gender, N-x1.shape[0])

    X_sel = np.concatenate([x1,x2])
    Y_sel = np.concatenate([y1,y2])
    feature = X_sel.shape[1] #number of features
    if use_model=='lr':
        perturb = np.random.randn(1, feature) / E  # perturbation matrix
        info_model = [None]
    elif use_model=='mlp':
        hidden_size, learning_rate, num_epochs, batch_size=128, 0.001, 400, 1000
        info_model = [hidden_size, learning_rate, num_epochs, batch_size]
        perturb = np.random.randn(1, hidden_size) / E # for the last layer



    all_accps = []
    all_accneg = []
    Pv_mcnemar = []
    Pv_delong = []
    for rp in range(S):
        # stratified k-fold cross validation
        kf = StratifiedKFold(n_splits=K, shuffle=True)

        # Lists to store training and validation indices
        train_indices = []
        val_indices = []

        # Split the data using StratifiedKFold
        for train_index, val_index in kf.split(X_sel, Y_sel):
            train_indices.append(train_index)
            val_indices.append(val_index)
        dist_p = '/datos/Bahram/classifiers/{}_{}/{}'.format(dataset_name, use_model, "_".join([str(el) for el in info]))
        if not os.path.exists(dist_p):
            os.makedirs(dist_p)

        with h5py.File('{}/XY_{}.h5'.format(dist_p, rp), 'w') as file:
            # Create datasets and write data to them
            file.create_dataset('X_sel', data=X_sel)
            file.create_dataset('Y_sel', data=Y_sel)
            file.create_dataset('train_indices', data=train_indices)
            file.create_dataset('val_indices', data=val_indices)
        """
        for reading
        with h5py.File('{}/XY.h5'.format(dist_p), 'r') as file:
            X_sel = np.array(file['X_sel'])
            Y_sel = np.array(file['Y_sel'])
            train_indices = np.array(file['train_indices'])
            val_indices = np.array(file['val_indices'])
        """
        accpos, accneg, [y_predp, y_predn, y_test] = cross_validation(X_sel, Y_sel, [train_indices, val_indices], perturb, info_model, use_model=use_model, early_break=False, dist_p=dist_p, repeat_no = rp)
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

    # perform corrected t_test
    p_value_corrected_t_test = corrected_t_test(all_accps, all_accneg, N, K)

    return [np.nanmean(Pv_mcnemar), np.nanmean(Pv_delong), p_value_t_test, p_value_corrected_t_test]





if __name__ == '__main__':

    global inputData, labels, device_id, dataset_name, use_model

    dataset_name = sys.argv[1]
    device_id=int(sys.argv[2])
    inputData, labels = get_data(dataset_name)
    use_model = 'mlp'

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
                        list_total.append([k, e, s, n, p])




    #for el in list_total:
    #    run_experiment(el)

    pool = mp.Pool(int(mp.cpu_count() //7))
    results = pool.map(run_experiment, list_total)
    dictionary = defaultdict(list)
    for i, el in enumerate(list_total):
        dictionary[i] = [el, results[i]]
    # write the results to a file
    with open('{}_perturbation_{}_sample_{}_model_{}.pkl'.format(dataset_name, Es[0], Ns[0], use_model), 'wb') as file:
        pickle.dump(dictionary, file)
