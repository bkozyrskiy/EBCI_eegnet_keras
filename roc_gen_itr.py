
from src.my_callbacks import AucMetricHistory
from src.models import EEGNet_old
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import sys
sys.path.append(os.path.join(os.path.split(os.getcwd())[0],'data_loader'))

# from data import DataBuildClassifier, OldData
from data import DataBuildClassifier
import numpy as np
import pickle
from keras.models import load_model
from sklearn.metrics import roc_auc_score,roc_curve
import shutil
from keras.utils import to_categorical
from src.utils import single_auc_loging
import codecs

def cv_test(x,y,model,model_path):
    model.save_weights('tmp.h5') # Nasty hack. This weights will be used to reset model

    best_val_epochs = []
    best_val_aucs = []

    folds = 4  # To preserve split as 0.6 0.2 0.2
    x_tr_ind, x_tst_ind, y_tr, y_tst = train_test_split(range(x.shape[0]), y, test_size=0.2, stratify=y)
    x_tr, x_tst = x[x_tr_ind], x[x_tst_ind]
    cv = StratifiedKFold(n_splits=folds,shuffle=True)
    cv_splits = list(cv.split(x_tr, y_tr[:,1]))

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        fold_model_path = os.path.join(model_path,'%d' % fold)
        os.makedirs(fold_model_path)
        same_subj_auc = AucMetricHistory(save_best_by_auc=True, path_to_save=fold_model_path)
        model.load_weights('tmp.h5') # Rest model on each fold
        x_tr_fold,y_tr_fold = x_tr[train_idx],y_tr[train_idx]
        x_val_fold, y_val_fold = x_tr[val_idx], y_tr[val_idx]
        val_history = model.fit(x_tr_fold, y_tr_fold, epochs=200, validation_data=(x_val_fold, y_val_fold),
                            callbacks=[same_subj_auc], batch_size=64, shuffle=True)
        best_val_epochs.append(np.argmax(val_history.history['val_auc']) + 1) # epochs count from 1 (not from 0)
        best_val_aucs.append(np.max(val_history.history['val_auc']))



    #Test  performance (Naive, until best epoch
    model.load_weights('tmp.h5') # Rest model before traning on train+val
    auc_history_naive = AucMetricHistory()
    test_history = model.fit(x_tr, y_tr, epochs=int(np.mean(best_val_epochs)),
                        validation_data=(x_tst, y_tst),callbacks=[auc_history_naive],batch_size=64, shuffle=True)

    naive_pred = model.predict(x_tst)


    model.save(os.path.join(model_path,'final_%d.hdf5' %int(np.mean(best_val_epochs))))
    with open(os.path.join(model_path,'testing_data.pkl'), 'wb') as output:
        pickle.dump((x_tst, y_tst),output,pickle.HIGHEST_PROTOCOL)

    os.remove('tmp.h5')

    # Test  performance (ensemble)
    best_models = []
    ensmbl_pred = np.zeros_like(y_tst)
    for fold_folder in os.listdir(model_path):
        fold_model_path = os.path.join(model_path,fold_folder)
        if os.path.isdir(fold_model_path):
            model_checkpoint = os.listdir(fold_model_path)[0]
            fold_model_path = os.path.join(fold_model_path,model_checkpoint)
            # best_models.append(load_model(fold_model_path))
            ensmbl_pred+=np.squeeze(load_model(fold_model_path).predict(x_tst))

    ensmbl_pred /= (folds)
    test_auc_ensemble = roc_auc_score(y_tst,ensmbl_pred)


    return np.mean(best_val_aucs),np.std(best_val_aucs), test_history.history,test_auc_ensemble, y_tst,naive_pred,ensmbl_pred


if __name__=='__main__':
    params = {'resample_to': 323,
             'regRate': 0,
             'dropoutRate0': 0.72,
             'dropoutRate1': 0.32,
             'dropoutRate2': 0.05,
             'lr': 0.0009
             }

    random_state=42
    data = DataBuildClassifier('/home/likan_blk/BCI/NewData')
    all_subjects = [25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38]
    experiment_res_dir = './res/cv_simple_ebci_roc_curves/'
    subjects = data.get_data(all_subjects,shuffle=False, windows=[(0.2,0.5)],baseline_window=(0.2, 0.3),resample_to=323)
    dropouts = (0.72, 0.32, 0.05)
    mean_val_aucs = []
    test_aucs_naive = []
    test_aucs_ensemble = []
    subjects_sets = all_subjects
    for train_subject in subjects_sets:
        path_to_save = os.path.join(experiment_res_dir, train_subject)
        model_path = os.path.join(path_to_save, 'checkpoints')
        if os.path.isdir(path_to_save):
            shutil.rmtree(path_to_save)
        os.makedirs(model_path)
        x = subjects[train_subject][0]
        y = to_categorical(subjects[train_subject][1], 2)

        model = EEGNet_old(params, Chans=x.shape[2], Samples=x.shape[1])
        x = x.transpose(0,2,1)[:,np.newaxis,:,:]
        mean_val_auc, std_val_auc, test_histpory, test_auc_ensemble,y_tst,naive_pred,ensmbl_pred = \
            cv_test(x, y, model, model_path)

        hyperparam_name = 'DO_%s' %('_'.join([str(dropout) for dropout in dropouts]))
        plot_name = '%s_%.02f_%d' %(hyperparam_name,test_histpory['val_auc'][-1],len(test_histpory['val_auc']))
        single_auc_loging(test_histpory, plot_name, path_to_save=path_to_save)
        with codecs.open('%s/res.txt' %path_to_save,'w', encoding='utf8') as f:
            f.write(u'Val auc %.02f±%.02f\n' %(mean_val_auc,std_val_auc))
            f.write('Test auc naive %.02f\n' % (test_histpory['val_auc'][-1]))
            f.write('Test auc ensemble %.02f\n' % test_auc_ensemble)
