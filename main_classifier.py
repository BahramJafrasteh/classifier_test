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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
import compare_auc_delong_xu
import sys
from MLP_model import MLPClassifier, train_perturb_evaluate_MLP_model, calc_accuracy
from sklearn.decomposition import PCA
import pandas as pd

def get_data(dataset_name):
    """
    Preparing the data to introduce them into the model
    :param dataset_name: name of the dataset
    :return: X (features) and y (labels)
    """
    if dataset_name.lower()=='abcd5':


        df_thk = pd.read_csv('abcd-data-release-5.0/core/imaging/mri_y_smr_thk_dsk.csv')
        df_area = pd.read_csv('abcd-data-release-5.0/core/imaging/mri_y_smr_area_dsk.csv')
        df_aseg = pd.read_csv('abcd-data-release-5.0/core/imaging/mri_y_smr_vol_aseg.csv')
        df_demo = pd.read_csv('abcd-data-release-5.0/core/abcd-general/abcd_p_demo.csv')

        sub_df_thk = df_thk[df_thk['eventname']=='baseline_year_1_arm_1']
        sub_df_area = df_area[df_area['eventname']=='baseline_year_1_arm_1']
        sub_df_aseg = df_aseg[df_aseg['eventname']=='baseline_year_1_arm_1']
        sub_df_demo = df_demo[df_demo['eventname']=='baseline_year_1_arm_1']


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
        #'ST78SV', 'ST19SV'


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
    elif dataset_name.lower()== 'hpc':

        out_come = ['Language_Task_Acc','Language_Task_Median_RT']
        nc = 25#15 or 25
        generate_data = False

        if generate_data:
            if nc == 15:
                folder_data ='HCP/node_timeseries/3T_HCP1200_MSMAll_d15_ts2/'
            else:
                folder_data ='HCP/node_timeseries/3T_HCP1200_MSMAll_d25_ts2/'
            df_head = pd.read_csv('HCP/unrestricted_qingyuz_10_5_2023_15_43_8.csv')
            df_head = df_head.dropna(subset=[out_come[0]])
            df_head = df_head.dropna(subset=[out_come[1]])
            mapping = {'M': 1, 'F': 0}
            df_head['Gender'] = df_head['Gender'].map(mapping)
            inpd = []
            for sb in df_head['Subject']:
                s_df = df_head[df_head['Subject']==sb]
                file = os.path.join(folder_data, str(sb)+'.txt')
                if os.path.isfile(file):
                    with open(file, 'rb') as f:
                        L = f.readlines()
                        arr = np.stack([np.array(list(map(float, e.decode('utf8').strip().split()))) for e in L])
                        corrM = np.corrcoef(arr, rowvar=False)
                        upper_triangle = corrM[np.triu_indices_from(corrM, k=1)]
                        conc = np.concatenate([upper_triangle, s_df['Gender'].values, s_df[out_come[0]].values, s_df[out_come[1]].values])
                        inpd.append(conc)
            xy = np.stack(inpd)
            df_final_xy = pd.DataFrame(xy)
            cols = ['var_' + str(e) for e in range(len(df_final_xy.columns))]
            cols[-1] = out_come[1]
            cols[-2] = out_come[0]
            cols[-3]= 'Gender'
            df_final_xy.columns = cols
            df_final_xy.to_csv('HCP/final_xy_{}.csv'.format(nc), index=None, index_label=None)
        df_final_xy = pd.read_csv('HCP/final_xy_{}.csv'.format(nc))
        cols = ['var_' + str(e) for e in range(len(df_final_xy.columns))]
        cols[-1] = out_come[1]
        cols[-2] = out_come[0]
        cols[-3] = 'Gender'
        inputData = df_final_xy[cols[:-2]].values
        labels = df_final_xy[out_come[0]].values
    elif dataset_name.lower() == 'abide':
        generate_data = False
        nc = 200
        if generate_data:
            if nc==116:
                abide_directory = 'abide/Outputs/cpac/nofilt_noglobal/rois_aal'
            elif nc==200:
                abide_directory = 'abide/cpac/nofilt_noglobal/rois_cc200'
            file_info = 'abide/Phenotypic_V1_0b_preprocessed1.csv'
            df_head = pd.read_csv(file_info)
            list_f = os.listdir(abide_directory)
            list_f = [f for f in list_f if f[-2:].lower()=='1d']
            inpd = []

            for file in list_f:
                s_df = df_head[df_head['FILE_ID'] == file.split('_rois')[0]]
                with open(os.path.join(abide_directory, file), 'rb') as f:
                    L = f.readlines()
                arr = np.stack([np.array(list(map(float, e.decode('utf8').strip().split()))) for e in L[1:]])
                # Check for constant columns
                constant_columns = np.all(np.diff(arr, axis=0) == 0, axis=0)
                if constant_columns.sum()>0:#if there is a constant row pass the id
                    continue

                corrM = np.corrcoef(arr, rowvar=False)
                upper_triangle = corrM[np.triu_indices_from(corrM, k=1)]

                conc = np.concatenate(
                    [upper_triangle, s_df['AGE_AT_SCAN'].values, s_df['SEX'].values-1,
                    s_df['DX_GROUP'].values])
                inpd.append(conc)
            xy = np.stack(inpd)
            df_final_xy = pd.DataFrame(xy)
            cols = ['var_' + str(e) for e in range(len(df_final_xy.columns))]
            cols[-1] = 'DX_GROUP'
            cols[-2] = 'SEX'
            cols[-3] = 'AGE_AT_SCAN'
            df_final_xy.columns = cols
            df_final_xy.to_csv('abide/final_xy_{}.csv'.format(nc), index=None, index_label=None)

        df_final_xy = pd.read_csv('abide/final_xy_{}.csv'.format(nc))
        cols = list(df_final_xy.columns)
        inputData = df_final_xy[cols[:-1]].values
        labels = df_final_xy['DX_GROUP'].values-1

    return inputData, labels

def McNemar(y_predp, y_predn, y_test):
    """
    compute the p-value of McNemar test
    :param y_predp: prediction of positively perturbed model
    :param y_predn: prediction of negatively perturbed model
    :param y_test: observed values
    :return: p-value of the test
    """
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
    """
    compute the p-value of DeLong test
    :param y_predp: prediction of positively perturbed model
    :param y_predn: prediction of negatively perturbed model
    :param y_test: observed values
    :return: p-value of the test
    """
    # Perform DeLong's test
    log_p = compare_auc_delong_xu.delong_roc_test(
        y_test.squeeze().astype('int'), y_predn.squeeze(), y_predp.squeeze())
    return np.exp(log_p).item()

def t_test(accp , accn):
    """
    compute the p-value of the Uncorrected t-test
    :param accp: accuracy of positively perturbed model
    :param accn:  accuracy of negatively perturbed model
    :return: p-value of the t-test
    """
    from scipy.stats import ttest_rel
    _, p_value = ttest_rel(accp, accn)
    return p_value



def corrected_t_test(accp , accn, N, K):
    """
    compute the p-value of the Corrected t-test
    :param accp: accuracy of positively perturbed model
    :param accn: accuracy of negatively perturbed model
    :param N: total number of dataset
    :param K: number of folds
    :return: p-value of the corrected t-test
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

def cross_validation(Xx, Yy, train_val_ind, perturb, info_model, use_model=None, early_break=False, dist_p=None, repeat_no=0):
    """
    Performing K-fold cross validaiton
    :param Xx: input features
    :param Yy: label
    :param train_val_ind: indices of training and validation for different KF
    :param perturb: perturbation vectors
    :param info_model: information of the model
    :param use_model: 'lr' or 'mlp'
    :param early_break:
    :param dist_p: direction to save
    :param repeat_no: index of repetition of CV
    :return:
    """


    if len(perturb)> 1:
        perturb1, perturb2 = perturb # for two completely random perturbation
    else:
        perturb1 = perturb[0]
        perturb2 = None
    accpos = []
    accneg = []
    accs = []
    accs_train = []

    [train_indices, val_indices] = train_val_ind
    if use_model=='mlp':
        hidden_size, learning_rate, num_epochs, batch_size = info_model
        device = torch.device('cpu') # use cpu (one can use gpu too)

        if dataset_name.lower()!='none':
            Xx = torch.from_numpy(Xx.astype(np.float32)).to(device)
            Yy = torch.from_numpy(Yy.astype(np.float32)).to(device)
        else:
            Yy = torch.from_numpy(Yy.astype(np.float32)).to(device)
        perturb1 = torch.from_numpy(perturb1.astype(np.float32)).to(device)
        if perturb2 is not None:
            perturb2 = torch.from_numpy(perturb2.astype(np.float32)).to(device)

        input_size = Xx.shape[1]

        output_size = 1
        model = MLPClassifier(input_size, hidden_size, output_size)
        model.to(device)

    y_predp_all = []
    y_predn_all = []
    y_test_all = []

    # do the CV
    for r, [train_index, test_index] in enumerate(zip(*[train_indices, val_indices])):

        X_train, X_test = Xx[train_index, :], Xx[test_index, :]
        y_train, y_test = Yy[train_index], Yy[test_index]


        if use_model=='lr':

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            forced = False
            out_f_name = os.path.join(dist_p, '{}_cv_{:03d}_rep_{}.pkl'.format(use_model, r + 1, repeat_no))
            if not os.path.isfile(out_f_name) or forced:
                model = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)
                model.fit(X_train, y_train.squeeze())
                with open(out_f_name, 'wb') as file:
                    pickle.dump(model, file)
            else:
                ### for reading
                with open(out_f_name, 'rb') as file:
                    model = pickle.load(file)



            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            coef_ = model.coef_.copy()


            if perturb2 is not None: # model perturbation
                coefp = coef_+perturb1
                coefn = coef_+perturb2
            else:
                coefp = coef_ + perturb1
                coefn = coef_ - perturb1

            model.coef_ = coefp
            y_predp = model.predict(X_test)
            model.coef_ = coefn
            y_predn = model.predict(X_test)

            y_predp_all.append(y_predp)
            y_predn_all.append(y_predn)
            y_test_all.append(y_test)

            acc = accuracy_score(y_pred, y_test)
            acc_train = accuracy_score(y_pred_train, y_train)

            accp = accuracy_score(y_predp, y_test)
            accn = accuracy_score(y_predn, y_test)

        elif use_model=='mlp':
            class_weights = y_train.shape[0] / (2 * torch.bincount(y_train.to(torch.int))[1])
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            out_f_name = os.path.join(dist_p, '{}_cv_{:03d}_rep_{}.pth'.format(use_model, r + 1, repeat_no))
            # Calculate mean and standard deviation along the specified axis (usually axis=0)
            mean = X_train.mean(0)
            std = X_train.std(0)+1e-10
            # normalize data
            X_train = (X_train-mean)/(std)
            X_test = (X_test-mean)/(std)
            model = MLPClassifier(input_size, hidden_size, output_size)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            y_pred, y_predp, y_predn , y_pred_train= train_perturb_evaluate_MLP_model(X_train,y_train, [perturb1,perturb2], X_test, model, optimizer, criterion,
                           num_epochs=num_epochs,    batch_size = batch_size, device=device, out_f_name=out_f_name, perturb_mode=perturb_mode,
                                                                        forced=False)

            acc = calc_accuracy(y_pred, y_test)
            acc_train = calc_accuracy(y_pred_train, y_train)

            accp = calc_accuracy(y_predp, y_test)
            accn = calc_accuracy(y_predn, y_test)

            y_predp_all.append(y_predp)
            y_predn_all.append(y_predn)
            y_test_all.append(y_test)


        accpos.append(accp)
        accneg.append(accn)
        accs.append(acc)
        accs_train.append(acc_train)

        if use_model=='mlp':
            y_predn_all = torch.cat(y_predn_all,dim=0).ravel().detach().cpu().numpy()
            y_predp_all = torch.cat(y_predp_all,dim=0).ravel().detach().cpu().numpy()
            y_test_all = torch.cat(y_test_all,dim=0).ravel().detach().cpu().numpy()
        else:
            y_predn_all = np.concatenate(y_predn_all,axis=0).ravel()
            y_predp_all = np.concatenate(y_predp_all, axis=0).ravel()
            y_test_all = np.concatenate(y_test_all, axis=0).ravel()
        return accpos, accneg, accs,accs_train, [y_predp_all, y_predn_all, y_test_all]

def select_sample(ind_ones, size):
    """
    Randomly select samples
    :param ind_ones:
    :param size:
    :return:
    """
    inp1 = inputData[ind_ones, :]
    lb1 = labels[ind_ones]
    if size< lb1.shape[0]:
        randomindinces_p = np.random.choice((ind_ones).sum(),size )
        X_sel = inp1[randomindinces_p, :]
        Y_sel = lb1[randomindinces_p]
    else:
        X_sel = inp1
        Y_sel = lb1

    return X_sel, Y_sel
def run_experiment(info):
    """
    run the main experiment
    :param info: K number of folds in cross-validation, E perturbation level, N number of the data and # S number of repetition of the K-fold cross-validation
    :return:
    """
    K, E, S, N, p =info

    ind_ones = (labels==1)
    x1,y1=select_sample(ind_ones, N//2)  # select the samples
    x2, y2=select_sample(~ind_ones, N-x1.shape[0])
    info2 = [el for el in info]
    info2[1] = 5 # pute info2[1] as 5 just for saving purpose
    dist_p = '{}_{}/{}'.format(dataset_name, use_model, "_".join([str(el) for el in info2]))
    if not os.path.exists(dist_p):
        os.makedirs(dist_p)
    X_sel = np.concatenate([x1,x2])
    Y_sel = np.concatenate([y1,y2])
    if dataset_name.lower()=='abide':
        # specific procedure for abide dataset
        file_int = '{}/XY_{}.pkl'.format(dist_p, 0)
        if os.path.isfile(file_int):
            with open(file_int,'rb') as file:
                [X_sel, Y_sel, _, _] = pickle.load(file)
        else:
            n_components = 256
            pca = PCA(n_components=n_components)
            mean_x = X_sel.mean(0)
            std_x = X_sel.std(0)
            X_sel = pca.fit_transform((X_sel - mean_x) / std_x)

    feature = X_sel.shape[1] #number of features
    if use_model=='lr':
        if perturb_mode=='gaussian': # for gaussian perturbation
            perturb = np.random.randn(1, feature) / E  # perturbation matrix
            perturb2 = None
        elif perturb_mode=='uniform': # for uniform perturbation
            perturb_factor = 1./ E
            perturb = (2 * np.random.rand(*(1, feature)) - 1)* perturb_factor  # perturbation matrix
        info_model = [None]
    elif use_model=='mlp':
        if dataset_name.lower()!='abide':
            hidden_size, learning_rate, num_epochs, batch_size=128, 0.001, 400, 100
        else:
            hidden_size, learning_rate, num_epochs, batch_size=128, 0.001, 400, 100
        info_model = [hidden_size, learning_rate, num_epochs, batch_size]
        if perturb_mode=='gaussian':
            perturb = np.random.randn(1, hidden_size) / E  # for the last layer
            perturb2 = None
            # for two completely rando models
            ##perturb = np.random.randn(1, hidden_size) / E  # for the last layer
            ##perturb2 = np.random.randn(1, hidden_size) / E  # for the last layer
        elif perturb_mode=='uniform':
            perturb_factor = 1./ E
            perturb = (2 * np.random.rand(*(1, hidden_size)) - 1)* perturb_factor   # perturbation matrix




    all_accps = []
    all_accneg = []
    Pv_mcnemar = []
    Pv_delong = []
    all_acc_non_perturb =[]
    all_acc_non_perturb_train =[]
    for rp in range(S): # repeat CV for specific number of times

        kf = StratifiedKFold(n_splits=K, shuffle=True) # stratified k-fold cross validation

        # Lists to store training and validation indices
        train_indices = []
        val_indices = []

        # Split the data using StratifiedKFold
        for train_index, val_index in kf.split(X_sel, Y_sel):
            train_indices.append(train_index)
            val_indices.append(val_index)

        file_split = '{}/XY_{}.pkl'.format(dist_p, rp)
        if not os.path.isfile(file_split):
            with open(file_split,'wb') as file:
                pickle.dump([X_sel, Y_sel, train_indices, val_indices], file)
        else:
            ### for reading
            with open(file_split,'rb') as file:
                [X_sel, Y_sel, train_indices, val_indices] = pickle.load(file)
        # di the cross validation
        accpos, accneg, acc_non_perturb,acc_train,  [y_predp, y_predn, y_test] = cross_validation(X_sel, Y_sel, [train_indices, val_indices], [perturb,perturb2], info_model, use_model=use_model, early_break=False, dist_p=dist_p, repeat_no = rp)

        # perfrom McNemar test
        pv_mcnemar = McNemar(y_predp, y_predn, y_test)
        Pv_mcnemar.append(pv_mcnemar)
        # Perform DeLong's test
        try:
            pvalue_delong = DeLong(y_predp, y_predn, y_test)
            Pv_delong.append(pvalue_delong)
        except:
            pass
        all_accps.append(np.array(accpos))
        all_accneg.append(np.array(accneg))
        all_acc_non_perturb.append(np.array(acc_non_perturb))
        all_acc_non_perturb_train.append(np.array(acc_train))
    all_accps = np.stack(all_accps).ravel()
    all_accneg = np.stack(all_accneg).ravel()
    all_acc_non_perturb = np.stack(all_acc_non_perturb).ravel()
    all_acc_non_perturb_train = np.stack(all_acc_non_perturb_train).ravel()

    # perform t_test
    p_value_t_test = t_test(all_accps, all_accneg)
    if p_value_t_test<0.05:
        print(p_value_t_test)
    # perform corrected t_test
    p_value_corrected_t_test = corrected_t_test(all_accps, all_accneg, N, K)


    return [np.nanmean(Pv_mcnemar), np.nanmean(Pv_delong), p_value_t_test, p_value_corrected_t_test, all_accps, all_accneg, all_acc_non_perturb_train, all_acc_non_perturb]





if __name__ == '__main__':

    global inputData, labels, dataset_name, use_model, perturb_mode


    dataset_name = sys.argv[1] # abide, abcd5, adni
    use_model = sys.argv[2] # 'mlp' or 'lr'
    P = sys.argv[3]  # repeat the whole experiment P times (default 100)


    inputData, labels = get_data(dataset_name)

    perturb_mode = 'gaussian' # The perturbation mode that can be gaussian or uniform

    Es = [1,2,3,4,5,6] # Perturbation level

    Ns = [1000]
    if dataset_name.lower()=='abide': # define the number of dataset for each dataset
        Ns = [600]
    if dataset_name.lower()=='adni':
        Ns = [444]

    Ks = [2, 5, 10, 25, 50] # define different number of K fold cross validation
    Ss = [1, 2, 4, 6, 10] # The number of repetition of the whole process


    if dataset_name.lower()=='adni':

        Ks = [2, 5, 10, 25, 50]
        Ss = [1, 2, 4, 6, 10]

    list_total = []
    for p in range(P):
        for k in Ks:
            for s in Ss:
                for e in Es: 
                    for n in Ns:
                        list_total.append([k, e, s, n, p])



    #####################################################
    pool = mp.Pool(1)#7
    results = pool.map(run_experiment, list_total)
    dictionary = defaultdict(list)
    for i, el in enumerate(list_total):
        dictionary[i] = [el, results[i]]

    average_training_acc = np.mean([dictionary[key][-1][-2].mean() for key in dictionary.keys()])
    average_test_acc = np.mean([dictionary[key][-1][-1].mean() for key in dictionary.keys()])
    print('AVERAGE TRAINING ACC: {}, AVERAGE TEST ACC: {}'.format(average_training_acc, average_test_acc))



    file_out = '{}_perturbation_{}_sample_{}_model_{}_pert_{}_numk_{}_nums_{}_randomperturbation.pkl'.format(dataset_name,
                                                                                          Es[0], Ns[0],
                                                                                          use_model,
                                                                                          perturb_mode,
                                                                                          len(Ks), len(Ss))
    print(file_out)
    with open(file_out, 'wb') as file:
        pickle.dump(dictionary, file)
