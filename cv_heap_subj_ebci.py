from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import codecs
import shutil

import os
import sys
sys.path.append(os.path.join(os.path.split(os.getcwd())[0],'ebci_data_loader'))

from data import DataBuildClassifier
# from data_proj_pursuit import DataProjPursuit
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import keras.backend as K
from src.NN import get_model
from src.utils import single_auc_loging, clean_bad_auc_models,set_seed
from src.my_callbacks import PerSubjAucMetricHistory,AucMetricHistory
from src.models import EEGNet
import numpy as np
import pickle
K.set_image_data_format("channels_first")

def heap_subj_test(subjects,model,model_path,train_subjects,val_subjects):
    # train_subj = [subj_idx for subj_idx in subjects.keys() if subj_idx not in test_subjects]
    X_train = np.concatenate([subjects[subj_idx][0] for subj_idx in train_subjects],axis=0)
    y_train = np.concatenate([subjects[subj_idx][1] for subj_idx in train_subjects],axis=0)

    X_val = np.concatenate([subjects[subj_idx][0] for subj_idx in val_subjects], axis=0)
    y_val = np.concatenate([subjects[subj_idx][1] for subj_idx in val_subjects], axis=0)
    auc_history =AucMetricHistory(save_best_by_auc=True,path_to_save=model_path)

    X_train = X_train.transpose(0, 2, 1)[:, np.newaxis, :, :]
    X_val = X_val.transpose(0, 2, 1)[:, np.newaxis, :, :]
    y_train = to_categorical(y_train,2)
    y_val = to_categorical(y_val, 2)

    val_history = model.fit(X_train, y_train, epochs=2, validation_data=(X_val, y_val),
                            callbacks=[auc_history], batch_size=64, shuffle=True)
    return val_history






if __name__=='__main__':
    random_state = 0
    set_seed(seed_value=random_state)

    params_v4 = {'resample_to': 369,
                 'D': 3,
                 'F1': 12,
                 'dropoutRate1': 0.52,
                 'dropoutRate2': 0.36,
                 'lr': 0.00066,
                 'norm_rate': 0.2756199103746462,
                 'time_filter_lenght': 99
                 }

    experiment_res_dir = './res/cv_heap_subj_ebci/'





    data = DataBuildClassifier('/home/amplifier/common/ebci_data/NewData')
    all_subjects = [25, 27, 28, 29, 30, 33] #BE CAREFUL, the 32th is excluded

    subjects = data.get_data(all_subjects, shuffle=False, windows=[(0.2, 0.5)], baseline_window=(0.2, 0.3),
                             resample_to=params_v4['resample_to'])

    val_subjects = [28, 29]
    train_subjects = [25, 27, 30, 33]

    path_to_save = os.path.join(experiment_res_dir, '_'.join(map(str,val_subjects)))
    model_path = os.path.join(path_to_save, 'check_points')
    if os.path.isdir(path_to_save):
        shutil.rmtree(path_to_save)
    os.makedirs(model_path)

    Chans = subjects[all_subjects[0]][0].shape[2]
    Samples = subjects[all_subjects[0]][0].shape[1]
    model = EEGNet(params_v4, nb_classes=2, F2=params_v4['F1'] * params_v4['D'], Chans=Chans, Samples=Samples)

    history = heap_subj_test(subjects, model, model_path, train_subjects, val_subjects)
    single_auc_loging(history.history, 'Val=%s' %str(val_subjects), path_to_save)
