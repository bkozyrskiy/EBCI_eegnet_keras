# -*- coding: utf-8 -*-

#Test, where classifier trained and tested on same subject throw full CV procedure with train, val and test split
#Simple netwrok, data ebci


from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import codecs
import shutil

import os
import sys
sys.path.append(os.path.join(os.path.split(os.getcwd())[0],'data_loader'))

from data import DataBuildClassifier
# from data_proj_pursuit import DataProjPursuit
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from src.NN import get_model
from src.utils import single_auc_loging, clean_bad_auc_models,set_seed
from src.my_callbacks import PerSubjAucMetricHistory,AucMetricHistory
from src.models import EEGNet_old,EEGNet
import numpy as np
import pickle



def cv_per_subj_test(x,y,model,model_path,block_mode = False,print_fold_history=False):
    model.save_weights('tmp.h5') # Nasty hack. This weights will be used to reset model
    same_subj_auc = AucMetricHistory()



    best_val_epochs = []
    best_val_aucs = []

    folds = 4  # To preserve split as 0.6 0.2 0.2
    if block_mode:
        targ_indices = [ind for ind in range(len(y)) if y[ind,1] == 1]
        nontarg_indices = [ind for ind in range(len(y)) if y[ind,1] == 0]
        tst_ind = targ_indices[int(0.8*len(targ_indices)):] + nontarg_indices[int(0.8*len(nontarg_indices)):]




        x_tr, x_tst = x[targ_indices[:int(0.8*len(targ_indices))] + nontarg_indices[:int(0.8*len(nontarg_indices))]],x[tst_ind]
        y_tr, y_tst = y[targ_indices[:int(0.8*len(targ_indices))] + nontarg_indices[:int(0.8*len(nontarg_indices))]],y[tst_ind]

        targ_tr_ind = list(range(int(0.8*len(targ_indices))))
        nontarg_tr_ind = list(range(int(0.8*len(targ_indices)),int(0.8*len(targ_indices)) + int(0.8*len(nontarg_indices))))


        targ_sections = list(map(int,np.linspace(0,1,folds+1)*len(targ_tr_ind)))
        nontarg_sections = list(map(int, np.linspace(0, 1, folds + 1) * len(nontarg_tr_ind)))

        cv_splits=[]
        for ind in range(folds):

            cv_splits.append(
                (targ_tr_ind[targ_sections[0]:targ_sections[ind]] + targ_tr_ind[targ_sections[ind+1]:targ_sections[-1]] + \
                nontarg_tr_ind[nontarg_sections[0]:nontarg_sections[ind]] + nontarg_tr_ind[nontarg_sections[ind + 1]:nontarg_sections[-1]],
                targ_tr_ind[targ_sections[ind]:targ_sections[ind + 1]] + nontarg_tr_ind[nontarg_sections[ind]:nontarg_sections[ind + 1]])
            )

    else:
        x_tr_ind, x_tst_ind, y_tr, y_tst = train_test_split(range(x.shape[0]), y, test_size=0.2, stratify=y)
        x_tr, x_tst = x[x_tr_ind], x[x_tst_ind]
        cv = StratifiedKFold(n_splits=folds,shuffle=True)
        cv_splits = list(cv.split(x_tr, y_tr[:,1]))

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
    # for fold, (train_idx, val_idx) in enumerate(cv.split(x_tr, y_tr)):
        fold_model_path = os.path.join(model_path,'%d' % fold)
        os.makedirs(fold_model_path)
        make_checkpoint = ModelCheckpoint(os.path.join(fold_model_path, '{epoch:02d}.hdf5'),
                                          monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
        model.load_weights('tmp.h5') # Rest model on each fold
        x_tr_fold,y_tr_fold = x_tr[train_idx],y_tr[train_idx]
        x_val_fold, y_val_fold = x_tr[val_idx], y_tr[val_idx]
        val_history = model.fit(x_tr_fold, y_tr_fold, epochs=200, validation_data=(x_val_fold, y_val_fold),
                            callbacks=[same_subj_auc,make_checkpoint], batch_size=64, shuffle=True)

        best_val_epochs.append(np.argmax(val_history.history['val_auc']) + 1) # epochs count from 1 (not from 0)
        best_val_aucs.append(np.max(val_history.history['val_auc']))
        clean_bad_auc_models(fold_model_path, val_history.history)
        if print_fold_history:
            single_auc_loging(val_history.history, 'fold %d' % fold, fold_model_path)


    #Test  performance (Naive, until best epoch
    model.load_weights('tmp.h5') # Rest model before traning on train+val

    test_history = model.fit(x_tr, y_tr, epochs=int(np.mean(best_val_epochs)),
                        validation_data=(x_tst, y_tst),callbacks=[same_subj_auc],batch_size=64, shuffle=True)
    model.save(os.path.join(model_path,'final_%d.hdf5' %int(np.mean(best_val_epochs))))

    with open(os.path.join(model_path,'testing_data.pkl'), 'wb') as output:
        pickle.dump((x_tst, y_tst),output,pickle.HIGHEST_PROTOCOL)

    os.remove('tmp.h5')

    # Test  performance (ensemble)
    best_models = []
    predictions = np.zeros_like(y_tst)
    for fold_folder in os.listdir(model_path):
        fold_model_path = os.path.join(model_path,fold_folder)
        if os.path.isdir(fold_model_path):
            model_checkpoint = [elem for elem in os.listdir(fold_model_path) if elem.split('.')[-1] =='hdf5'][0]
            fold_model_path = os.path.join(fold_model_path,model_checkpoint)
            # best_models.append(load_model(fold_model_path))
            predictions+=np.squeeze(load_model(fold_model_path).predict(x_tst))

    predictions /= (folds)
    test_auc_ensemble = roc_auc_score(y_tst,predictions)


    return np.mean(best_val_aucs),np.std(best_val_aucs), test_history.history,test_auc_ensemble



if __name__=='__main__':
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

    data = DataBuildClassifier('/home/likan_blk/BCI/NewData')
    all_subjects = [25, 26,27,28,29,30,32,33,34,35,36,37,38]
    experiment_res_dir = './res/cv_simple_ebci/EEGNET_v%d/' %EEGNET_VERSION
    # data = OldData('/home/likan_blk/Yandex.Disk/e4/data', target_events=['L500BP','L500BC', 'R500BP','R500BC'],
    #                    nontarget_events=['L500BB', 'R500BB'])
    # all_subjects = range(8)

    # data = DataProjPursuit('/home/likan_blk/BCI/DataProjPursuit/')
    # all_subjects = ['subj%d' %i for i in range(1,7)]
    # experiment_res_dir = './res/cv_proj_pursuit/'

    subjects = data.get_data(all_subjects,shuffle=False, windows=[(0.2,0.5)],baseline_window=(0.2,0.3),resample_to=params['resample_to'])
    mean_val_aucs=[]
    test_aucs_naive = []
    test_aucs_ensemble = []
    subjects_sets = all_subjects
    for train_subject in subjects_sets:

        path_to_save = os.path.join(experiment_res_dir,str(train_subject))
        model_path = os.path.join(path_to_save,'checkpoints')
        if os.path.isdir(path_to_save):
            shutil.rmtree(path_to_save)
        os.makedirs(model_path)
        x = subjects[train_subject][0]
        y = to_categorical(subjects[train_subject][1],2)
        # y = subjects[train_subject][1]

        if EEGNET_VERSION==2:
            model = EEGNet_old(params_v2, Chans=x.shape[2], Samples=x.shape[1])

        if EEGNET_VERSION==4:
            model = EEGNet(params_v4,nb_classes=2,F2=params['F1']*params['D'], Chans=x.shape[2], Samples=x.shape[1])

        x = x.transpose(0,2,1)[:,np.newaxis,:,:]


        mean_val_auc,std_val_auc, test_histpory,test_auc_ensemble = cv_per_subj_test(x, y, model, model_path, block_mode = False, print_fold_history=True)

        dropouts = [params_v2[k] for k in params_v2.keys() if k.startswith('dropout') ]
        hyperparam_name = 'DO_%s' %('_'.join([str(dropout) for dropout in dropouts]))
        plot_name = '%s_%.02f_%d' %(hyperparam_name,test_histpory['val_auc'][-1],len(test_histpory['val_auc']))
        # if os.path.isdir(path_to_save):
        #     shutil.rmtree(path_to_save)
        single_auc_loging(test_histpory, plot_name, path_to_save=path_to_save)
        mean_val_aucs.append((mean_val_auc,std_val_auc))
        test_aucs_naive.append(test_histpory['val_auc'][-1])
        test_aucs_ensemble.append(test_auc_ensemble)
        with codecs.open('%s/res.txt' %path_to_save,'w', encoding='utf8') as f:
            f.write(u'Val auc %.02f±%.02f\n' %(mean_val_auc,std_val_auc))
            f.write('Test auc naive %.02f\n' % (test_histpory['val_auc'][-1]))
            f.write('Test auc ensemble %.02f\n' % test_auc_ensemble)

    with codecs.open('%s/res.txt' %experiment_res_dir,'w', encoding='utf8') as f:
        f.write('subj,mean_val_aucs,test_aucs_naive,test_aucs_ensemble\n')
        for tr_subj_idx, tr_subj in enumerate(subjects_sets):
            f.write(u'%s, %.02f±%.02f, %.02f, %.02f\n' \
                    % (tr_subj,mean_val_aucs[tr_subj_idx][0],mean_val_aucs[tr_subj_idx][1],test_aucs_naive[tr_subj_idx],
                       test_aucs_ensemble[tr_subj_idx]))

        final_val_auc = np.mean(list(zip(*mean_val_aucs))[0])
        final_auc_naive = np.mean(test_aucs_naive)
        final_auc_ensemble = np.mean(test_aucs_ensemble)
        final_val_auc_std = np.std(list(zip(*mean_val_aucs))[0])
        final_auc_naive_std = np.std(test_aucs_naive)
        final_auc_ensemble_std = np.std(test_aucs_ensemble)
        f.write(u'MEAN, %.02f±%.02f, %.02f±%.02f, %.02f±%.02f\n' \
                % (final_val_auc,final_val_auc_std,final_auc_naive,final_auc_naive_std,final_auc_ensemble,
                   final_auc_ensemble_std))


###############################################################################################################