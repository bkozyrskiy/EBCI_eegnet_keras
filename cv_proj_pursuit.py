from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from sklearn.metrics import roc_auc_score
import codecs
import shutil

import os
import sys
sys.path.append(os.path.join(os.path.split(os.getcwd())[0],'data_loader'))

# from data import DataBuildClassifier, OldData
from data_proj_pursuit_v2 import DataProjPursuit_v2
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import keras.backend as K
from src.NN import get_model
from src.utils import single_auc_loging, clean_bad_auc_models,find_files_by_ext,set_seed
from src.my_callbacks import PerSubjAucMetricHistory,AucMetricHistory
from src.models import EEGNet_old,EEGNet
import numpy as np
import pickle
from scipy.io import savemat


def cv_test(x,y,model,model_path,block_mode = False,test_data=None):
    model.save_weights('tmp.h5') # Nasty hack. This weights will be used to reset model
    same_subj_auc = AucMetricHistory()

    best_val_epochs = []
    best_val_aucs = []

    if test_data:
        x_tst,y_tst = test_data

    folds = 4  # To preserve split as 0.6 0.2 0.2
    if block_mode:
        targ_indices = [ind for ind in range(len(y)) if y[ind] == 1]
        nontarg_indices = [ind for ind in range(len(y)) if y[ind] == 0]
        if test_data is None:
            targ_indices_tr,targ_indices_tst = train_test_split(targ_indices,test_size=0.2,shuffle=False)
            nontarg_indices_tr, nontarg_indices_tst = train_test_split(nontarg_indices, test_size=0.2, shuffle=False)
            # x_tst = x[np.hstack((targ_indices_tst,nontarg_indices_tst))]
            # y_tst = y[np.hstack((targ_indices_tst, nontarg_indices_tst))]
        else:
            targ_indices_tr = targ_indices
            nontarg_indices_tr = nontarg_indices

        x_tr = x[np.hstack((targ_indices_tr, nontarg_indices_tr))]
        y_tr = y[np.hstack((targ_indices_tr, nontarg_indices_tr))]

        targ_indices_tr = np.where(y_tr == 1)[0]
        nontarg_indices_tr = np.where(y_tr == 0)[0]

        kf = KFold(n_splits = folds)
        f = lambda inds: (np.hstack((targ_indices_tr[inds[0][0]],nontarg_indices_tr[inds[1][0]])),
                          np.hstack((targ_indices_tr[inds[0][1]],nontarg_indices_tr[inds[1][1]])))
        cv_splits = map(f, zip(kf.split(targ_indices_tr),kf.split(nontarg_indices_tr)))
    else:
        # if test_data is None:
        #     x_tr_ind, x_tst_ind, y_tr, y_tst = train_test_split(range(x.shape[0]), y, test_size=0.2, stratify=y)
        #     x_tr, x_tst = x[x_tr_ind], x[x_tst_ind]
        # else:
        x_tr,y_tr = x,y

        cv = StratifiedKFold(n_splits=folds,shuffle=True)
        cv_splits = list(cv.split(x_tr, y_tr))

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
    # for fold, (train_idx, val_idx) in enumerate(cv.split(x_tr, y_tr)):
        fold_model_path = os.path.join(model_path,'%d' % fold)
        os.makedirs(fold_model_path)
        make_checkpoint = ModelCheckpoint(os.path.join(fold_model_path, '{epoch:02d}.hdf5'),
                                          monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
        model.load_weights('tmp.h5') # Rest model on each fold
        x_tr_fold,y_tr_fold = x_tr[train_idx],y_tr[train_idx]
        x_val_fold, y_val_fold = x_tr[val_idx], y_tr[val_idx]
        val_history = model.fit(x_tr_fold, to_categorical(y_tr_fold,2), epochs=200, validation_data=(x_val_fold, to_categorical(y_val_fold,2)),
                            callbacks=[same_subj_auc,make_checkpoint], batch_size=64, shuffle=True)
        best_val_epochs.append(np.argmax(val_history.history['val_auc']) + 1) # epochs count from 1 (not from 0)
        best_val_aucs.append(np.max(val_history.history['val_auc']))
        clean_bad_auc_models(fold_model_path, val_history.history)


    #Test  performance (Naive, until best epoch
    model.load_weights('tmp.h5') # Rest model before traning on train+val

    # test_history = model.fit(x_tr, to_categorical(y_tr,2), epochs=int(np.mean(best_val_epochs)),
    #                     validation_data=(x_tst, to_categorical(y_tst,2)),callbacks=[same_subj_auc],batch_size=64, shuffle=True)

    test_history = model.fit(x_tr, to_categorical(y_tr, 2), epochs=int(np.mean(best_val_epochs)),
                                                 validation_data=(None if test_data==None else (x_tst, to_categorical(y_tst,2))),callbacks=[same_subj_auc],batch_size=64, shuffle=True)
    model.save(os.path.join(model_path,'final_%d.hdf5' %int(np.mean(best_val_epochs))))

    # with open(os.path.join(model_path,'testing_data.pkl'), 'wb') as output:
    #     pickle.dump((x_tst, y_tst),output,pickle.HIGHEST_PROTOCOL)

    os.remove('tmp.h5')

    # Test  performance (ensemble)

    test_auc_ensemble= ensemble_evaluate(model_path,test_data)
    K.clear_session()
    return np.mean(best_val_aucs),np.std(best_val_aucs), test_history.history,test_auc_ensemble

def ensemble_evaluate(models_path,test_data):
    x_tst,y_tst = test_data
    predictions = np.zeros_like(y_tst, dtype=float)
    folds = 0
    for fold_folder in os.listdir(models_path):
        fold_model_path = os.path.join(models_path, fold_folder)
        if os.path.isdir(fold_model_path):
            model_checkpoint = os.listdir(fold_model_path)[0]
            fold_model_path = os.path.join(fold_model_path, model_checkpoint)
            # best_models.append(load_model(fold_model_path))
            predictions += np.squeeze(load_model(fold_model_path).predict(x_tst))[:, 1]
            folds += 1
    predictions /= (folds)
    test_auc_ensemble = roc_auc_score(y_tst, predictions)
    K.clear_session()
    return test_auc_ensemble

def naive_evaluate(model_path,test_data,model=None):
    x_tst,y_tst = test_data
    if model is None:
        predictions = load_model(model_path).predict(x_tst)[:,1]
    else:
        predictions = model.predict(x_tst)[:,1]
    test_auc_naive = roc_auc_score(y_tst, predictions)
    K.clear_session()
    return test_auc_naive



if __name__ == '__main__':
    random_state = 0
    set_seed(seed_value=random_state)
    EEGNET_VERSION=4
    params_v2 = {'resample_to': 323,
                 'regRate': 0,
                 'dropoutRate1': 0.72,
                 'dropoutRate2': 0.32,
                 'dropoutRate3': 0.05,
                 'lr': 0.0009
                 }

    params_v4 = {'resample_to': 369,
                 'D': 3,
                 'F1': 12,
                 'dropoutRate1': 0.52,
                 'dropoutRate2': 0.36,
                 'lr': 0.00066,
                 'norm_rate': 0.2756199103746462
                 }

    if EEGNET_VERSION == 4:
        params = params_v4
    elif EEGNET_VERSION == 2:
        params = params_v2
    data = DataProjPursuit_v2('/home/likan_blk/BCI/ProjPursuitData/')
    # all_subjects = list(range(1,15))
    all_subjects = list(range(1,15))
    experiment_res_dir = './res/cv_proj_pursuit/EEGNET_v%d/' %EEGNET_VERSION

    mean_val_aucs=[]
    total_test_aucs_naive = []
    total_test_aucs_ensemble = []
    cl1_cl3_test_aucs_naive = []
    cl1_cl3_test_aucs_ensemble = []
    cl4_cl3_test_aucs_naive = []
    cl4_cl3_test_aucs_ensemble = []

    subjects_sets = all_subjects
    for train_subject in subjects_sets:

        eeg_ch = range(19)
        cl3_eeg = data.get_event_data(train_subject, 'cl3', eeg_ch=eeg_ch,resample_to=params['resample_to'],
                                      window=(-0.3, 0), baseline_window=(-0.1, -0.0),shuffle=False)
        cl1_eeg = data.get_event_data(train_subject, 'cl1', eeg_ch=eeg_ch, resample_to=params['resample_to'], window=(-0.3, 0),
                                    baseline_window=(-0.1, -0.0),shuffle=False)
        cl4_eeg = data.get_event_data(train_subject, 'cl4', eeg_ch=eeg_ch, resample_to=params['resample_to'], window=(-0.3, 0),
                                    baseline_window=(-0.1, -0.0),shuffle=False)

        cl3_tr, cl3_tst = train_test_split(cl3_eeg, test_size=0.2,shuffle=False)
        cl1_tr,cl1_tst = train_test_split(cl1_eeg, test_size=0.2,shuffle=False)
        cl4_tr, cl4_tst = train_test_split(cl4_eeg, test_size=0.2,shuffle=False)
        y_tr = np.array([0]*cl3_tr.shape[0] + [1]*cl1_tr.shape[0] + [1]*cl4_tr.shape[0])
        x_tr = np.concatenate((cl3_tr, cl1_tr, cl4_tr),axis=0)

        y_tst_total = np.array([0]*cl3_tst.shape[0] + [1]*cl1_tst.shape[0] + [1]*cl4_tst.shape[0])
        x_tst_total = np.concatenate((cl3_tst, cl1_tst, cl4_tst),axis=0)

        y_tst_cl1_cl3 = np.array([1]*cl1_tst.shape[0] + [0]*cl3_tst.shape[0])
        x_tst_cl1_cl3 = np.concatenate((cl1_tst, cl3_tst),axis=0)
        y_tst_cl4_cl3 = np.array([1]*cl4_tst.shape[0] + [0]*cl3_tst.shape[0])
        x_tst_cl4_cl3 = np.concatenate((cl4_tst, cl3_tst),axis=0)


        if EEGNET_VERSION==2:
            model = EEGNet_old(params_v2, Chans=x_tr.shape[2], Samples=x_tr.shape[1])
        elif EEGNET_VERSION==4:
            model = EEGNet(params_v4,nb_classes=2,F2=params['F1']*params['D'], Chans=x_tr.shape[2], Samples=x_tr.shape[1])

        x_tr = x_tr.transpose(0, 2, 1)[:, np.newaxis, :, :]
        x_tst_total = x_tst_total.transpose(0, 2, 1)[:, np.newaxis, :, :]
        x_tst_cl1_cl3 = x_tst_cl1_cl3.transpose(0, 2, 1)[:, np.newaxis, :, :]
        x_tst_cl4_cl3 = x_tst_cl4_cl3.transpose(0, 2, 1)[:, np.newaxis, :, :]

        path_to_save = os.path.join(experiment_res_dir, str(train_subject))
        model_path = os.path.join(path_to_save, 'checkpoints')
        if os.path.isdir(path_to_save):
            shutil.rmtree(path_to_save)
        os.makedirs(model_path)

        # with open(os.path.join(model_path, 'cl3_cl1_cl4_tst.pkl'), 'wb') as output:
        #     pickle.dump((cl3_tst, cl1_tst,cl4_tst), output, pickle.HIGHEST_PROTOCOL)

        mean_val_auc, std_val_auc, test_histpory,test_auc_ensemble = cv_test(x_tr, y_tr, model, model_path, block_mode=False,test_data=(x_tst_total,y_tst_total))
        final_model_path = find_files_by_ext(model_path,'hdf5')
        test_auc_naive_cl1_cl3 = naive_evaluate(model_path=final_model_path,test_data=(x_tst_cl1_cl3,y_tst_cl1_cl3))
        test_auc_naive_cl4_cl3 = naive_evaluate(model_path=final_model_path,test_data=(x_tst_cl4_cl3,y_tst_cl4_cl3))

        test_auc_ens_cl1_cl3 = ensemble_evaluate(models_path=model_path,test_data=(x_tst_cl1_cl3,y_tst_cl1_cl3))
        test_auc_ens_cl4_cl3 = ensemble_evaluate(models_path=model_path,test_data=(x_tst_cl4_cl3,y_tst_cl4_cl3))

        # with open(os.path.join(model_path, 'eeg_raw.pkl'), 'wb') as output:
        #     pickle.dump((cl3_tst, cl1_tst,cl4_tst), output, pickle.HIGHEST_PROTOCOL)

        dropouts = [params[k] for k in params.keys() if k.startswith('dropout') ]
        hyperparam_name = 'DO_%s' %('_'.join([str(dropout) for dropout in dropouts]))
        plot_name = '%s_%.02f_%d' %(hyperparam_name,test_histpory['val_auc'][-1],len(test_histpory['val_auc']))
        # if os.path.isdir(path_to_save):
        #     shutil.rmtree(path_to_save)
        single_auc_loging(test_histpory, plot_name, path_to_save=path_to_save)
        mean_val_aucs.append((mean_val_auc,std_val_auc))
        total_test_aucs_naive.append(test_histpory['val_auc'][-1])
        total_test_aucs_ensemble.append(test_auc_ensemble)

        cl1_cl3_test_aucs_naive.append(test_auc_naive_cl1_cl3)
        cl1_cl3_test_aucs_ensemble.append(test_auc_ens_cl1_cl3)

        cl4_cl3_test_aucs_naive.append(test_auc_naive_cl4_cl3)
        cl4_cl3_test_aucs_ensemble.append(test_auc_ens_cl4_cl3)
        with codecs.open('%s/res.txt' %path_to_save,'w', encoding='utf8') as f:
            f.write(u'Val auc %.02f±%.02f\n' %(mean_val_auc,std_val_auc))
            f.write('Total test auc naive %.02f\n' % (test_histpory['val_auc'][-1]))
            f.write('Total test auc ensemble %.02f\n' % test_auc_ensemble)
            f.write('Cl1 vs cl3 test auc naive %.02f\n' % test_auc_naive_cl1_cl3)
            f.write('Cl1 vs cl3 test auc ensemble %.02f\n' % test_auc_ens_cl1_cl3)
            f.write('Cl4 vs cl3 test auc naive %.02f\n' % test_auc_naive_cl4_cl3)
            f.write('Cl4 vs cl3 test auc ensemble %.02f\n' % test_auc_ens_cl4_cl3)

    with codecs.open('%s/res.txt' %experiment_res_dir,'w', encoding='utf8') as f:
        f.write('subj,mean_val_aucs,total_test_aucs_naive,total_test_aucs_ensemble,cl1_cl3_tst_naive,cl1_cl3_tst_ens,cl4_cl3_tst_naive,cl4_cl3_tst_ens\n')
        for tr_subj_idx, tr_subj in enumerate(subjects_sets):
            f.write(u'%s, %.02f±%.02f, %.02f, %.02f, %.02f, %.02f, %.02f, %.02f\n' \
                    % (tr_subj,mean_val_aucs[tr_subj_idx][0],mean_val_aucs[tr_subj_idx][1],total_test_aucs_naive[tr_subj_idx],
                       total_test_aucs_ensemble[tr_subj_idx],cl1_cl3_test_aucs_naive[tr_subj_idx],cl1_cl3_test_aucs_ensemble[tr_subj_idx],
                       cl4_cl3_test_aucs_naive[tr_subj_idx],cl4_cl3_test_aucs_ensemble[tr_subj_idx])
                    )

        final_val_auc = np.mean(list(zip(*mean_val_aucs))[0])
        final_total_auc_naive = np.mean(total_test_aucs_naive)
        final_total_auc_ensemble = np.mean(total_test_aucs_ensemble)
        final_cl1_cl3_auc_naive = np.mean(total_test_aucs_naive)
        final_cl1_cl3_auc_ensemble = np.mean(total_test_aucs_ensemble)
        final_cl4_cl3_auc_naive = np.mean(total_test_aucs_naive)
        final_cl4_cl3_auc_ensemble = np.mean(total_test_aucs_ensemble)

        final_val_auc_std = np.std(list(zip(*mean_val_aucs))[0])
        final_total_auc_naive_std = np.std(total_test_aucs_naive)
        final_total_auc_ensemble_std = np.std(total_test_aucs_ensemble)
        final_cl1_cl3_auc_naive_std = np.std(total_test_aucs_naive)
        final_cl1_cl3_auc_ensemble_std = np.std(total_test_aucs_ensemble)
        final_cl4_cl3_auc_naive_std = np.std(total_test_aucs_naive)
        final_cl4_cl3_auc_ensemble_std = np.std(total_test_aucs_ensemble)
        f.write(u'MEAN, %.02f±%.02f, %.02f±%.02f, %.02f±%.02f, %.02f±%.02f, %.02f±%.02f, %.02f±%.02f, %.02f±%.02f\n' \
                % (final_val_auc,final_val_auc_std,final_total_auc_naive,final_total_auc_naive_std,final_total_auc_ensemble,
                   final_total_auc_ensemble_std,final_cl1_cl3_auc_naive,final_cl1_cl3_auc_naive_std,final_cl1_cl3_auc_ensemble,
                   final_cl1_cl3_auc_ensemble_std,final_cl4_cl3_auc_naive,final_cl4_cl3_auc_naive_std,final_cl4_cl3_auc_ensemble,
                   final_cl4_cl3_auc_ensemble_std))
