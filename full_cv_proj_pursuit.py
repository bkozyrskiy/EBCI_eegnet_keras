from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import codecs
import shutil

import os
import sys
sys.path.append(os.path.join(os.path.split(os.getcwd())[0],'data_loader'))

from data import DataBuildClassifier
from data_proj_pursuit import DataProjPursuit
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from src.NN import get_model
from src.utils import single_auc_loging, clean_bad_auc_models
from src.my_callbacks import PerSubjAucMetricHistory,AucMetricHistory
from src.models import EEGNet_old
import numpy as np
import pickle
from scipy.io import savemat
from cv_proj_pursuit import cv_test


def cv_test_full_set(x,y,net_params,res_path,folds=5):
    x_targ = x[y==1]
    x_nontarg = x[y==0]

    cv = KFold(n_splits=folds,shuffle=False)
    test_aucs_naive = []
    test_aucs_ensemble = []
    fold = -1
    for (targ_tr_ind,targ_tst_ind),(nontarg_tr_ind,nontarg_tst_ind) in zip(cv.split(x_targ),cv.split(x_nontarg)):

        fold += 1
        x_fold_tr = np.concatenate((x_nontarg[nontarg_tr_ind],x_targ[targ_tr_ind]),axis=0)
        y_fold_tr = np.concatenate((np.zeros_like(nontarg_tr_ind),np.ones_like(targ_tr_ind)),axis=0)

        x_fold_tst = np.concatenate((x_nontarg[nontarg_tst_ind],x_targ[targ_tst_ind]),axis=0)
        y_fold_tst = np.concatenate((np.zeros_like(nontarg_tst_ind),np.ones_like(targ_tst_ind)),axis=0)

        path_to_fold = os.path.join(res_path,'test_%d' %fold)
        fold_model_path = os.path.join(path_to_fold,'checkpoints')
        model = EEGNet_old(params, Chans=x.shape[2], Samples=x.shape[3])
        mean_val_auc, std_val_auc, fold_test_history, fold_test_auc_ensemble = cv_test(x_fold_tr,y_fold_tr,model,fold_model_path,block_mode = False,
                                                                  test_data = (x_fold_tst,y_fold_tst))


        dropouts = [net_params['dropoutRate%d' % layer] for layer in range(3)]
        hyperparam_name = 'DO_%s' % ('_'.join([str(dropout) for dropout in dropouts]))
        plot_name = '%s_%.02f_%d' % (hyperparam_name, fold_test_history['val_auc'][-1], len(fold_test_history['val_auc']))
        # if os.path.isdir(path_to_save):
        #     shutil.rmtree(path_to_save)
        single_auc_loging(fold_test_history, plot_name, path_to_save=path_to_fold)
        test_aucs_naive.append(fold_test_history['val_auc'][-1])
        test_aucs_ensemble.append(fold_test_auc_ensemble)
        with codecs.open('%s/fold_res.txt' % path_to_fold, 'w', encoding='utf8') as f:
            f.write(u'Val auc %.02f±%.02f\n' % (mean_val_auc, std_val_auc))
            f.write('Test auc naive %.02f\n' % (fold_test_history['val_auc'][-1]))
            f.write('Test auc ensemble %.02f\n' % fold_test_auc_ensemble)

    with codecs.open('%s/mean_res.txt' % res_path, 'w', encoding='utf8') as f:
        f.write('Test auc naive %.02f±%.02f\n' % (np.mean(test_aucs_naive),np.std(test_aucs_naive)))
        f.write('Test auc ensemble %.02f±%.02f\n' % (np.mean(test_aucs_ensemble),np.std(test_aucs_ensemble)))
    return np.mean(test_aucs_naive),np.std(test_aucs_naive),np.mean(test_aucs_ensemble),np.std(test_aucs_ensemble)

if __name__ == '__main__':
    params = {'resample_to': 323,
              'regRate': 0,
              'dropoutRate0': 0.72,
              'dropoutRate1': 0.32,
              'dropoutRate2': 0.05,
              'lr': 0.0009
              }
    random_state = 42
    data = DataProjPursuit('/home/likan_blk/BCI/DataProjPursuit/')
    # data = DataBuildClassifier('/home/likan_blk/BCI/NewData')
    # all_subjects = [25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38]
    # subjects = data.get_data(all_subjects, shuffle=False, windows=[(0.2, 0.5)],baseline_window=(0.2, 0.3), resample_to=323)
    train_subject = 1
    eeg_ch = range(19)
    experiment_res_dir = os.path.join(os.getcwd(), 'res/full_cv_proj_pursuit/')
    all_subjects = range(1,15)
    res = {}
    for train_subject in all_subjects:

        cl3_eeg = data.get_event_data(train_subject, 'cl3', eeg_ch=eeg_ch, rej_thrs=150, resample_to=None, window=(-0.4, 0),
                                      baseline_window=(-0.1, 0), shuffle=False)
        cl1_eeg = data.get_event_data(train_subject, 'cl1', eeg_ch=eeg_ch, rej_thrs=150, resample_to=None, window=(-0.4, 0),
                                      baseline_window=(-0.1, 0), shuffle=False)
        cl4_eeg = data.get_event_data(train_subject, 'cl1', eeg_ch=eeg_ch, rej_thrs=150, resample_to=None, window=(-0.4, 0),
                                      baseline_window=(-0.1, 0), shuffle=False)

        y = np.array([0]*cl3_eeg.shape[0] + [1]*cl1_eeg.shape[0])
        x = np.concatenate((cl3_eeg, cl1_eeg),axis=0)
        # x = subjects[train_subject][0]
        # y = subjects[train_subject][1]

        x = x.transpose(0, 2, 1)[:, np.newaxis, :, :]


        path_to_save = os.path.join(experiment_res_dir, str(train_subject))
        if os.path.isdir(path_to_save):
            shutil.rmtree(path_to_save)
        res[train_subject] = cv_test_full_set(x,y,params,path_to_save,folds=5)

    with codecs.open('%s/all_subj_res.txt' % experiment_res_dir, 'w', encoding='utf8') as f:
        f.write('subj,mean_val_aucs,test_aucs_naive,test_aucs_ensemble\n')
        for tr_subj in all_subjects:
            mean_auc_naive, std_auc_naive, mean_auc_ens, std_auc_ens = res[tr_subj]
            f.write(u'%s, %.02f±%.02f, %.02f±%.02f\n' \
                    % (
                        tr_subj, mean_auc_naive, std_auc_naive,
                        mean_auc_ens,std_auc_ens))