#Test, where group of subjects (group can we with single subj) used for training and all others for
#validating SEPARATELY. Simple netwrok, data ebci

import numpy as np
import shutil
import os
from src.data import DataBuildClassifier
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from src.NN import get_model
from src.utils import multi_auc_loging,single_auc_loging, multisubj_val_split,clean_bad_auc_models
from src.my_callbacks import PerSubjAucMetricHistory,AucMetricHistory
from src.feature_relevance import grads_wrt_input, apply_deep_expl
from sklearn.model_selection import train_test_split
import pickle
# from src.EEGModels import EEGNet



if __name__=='__main__':
    random_state=42
    data = DataBuildClassifier('/home/likan_blk/BCI/NewData')
    all_subjects = [25,26,27,28,29,30,32,33,34,35,36,37,38]
    subjects = data.get_data(all_subjects,shuffle=False, windows=[(0.2,0.5)],baseline_window=(0.2, 0.3))
    dropouts = (0.1,0.2,0.4)
    # subjects_sets = [(33, 34), (35, 36), (37, 38)]
    subjects_sets = [25,26,27,28,29,30,32,33,34,35,36,37,38]

    for train_subject in subjects_sets:
        path_to_save = './res/simple_ebci/%s/' % (train_subject)
        hyperparam_name = 'dropout_%s' % ('_'.join([str(dropout) for dropout in dropouts]))
        if os.path.isdir(path_to_save):
            shutil.rmtree(path_to_save)
        os.makedirs(path_to_save)
        model_path = os.path.join(path_to_save, 'checkpoints')
        os.makedirs(model_path)
        # checkpoint = ModelCheckpoint(os.path.join(model_path, '{epoch:02d}.hdf5'), monitor='val_loss', verbose=0, save_best_only=False, mode='auto')

        # x_train = np.concatenate([subjects[subj][0] for subj in train_subjects])
        # y_train = np.concatenate([subjects[subj][1] for subj in train_subjects])


        # x_train, x_val, y_train, y_val = train_test_split(subjects[train_subject][0], subjects[train_subject][1],
        #                                                   test_size = 0.2, stratify=subjects[train_subject][1])
        x,y = subjects[train_subject]
        train_ind, val_ind = train_test_split(range(len(y)), test_size = 0.2, stratify=y)
        x_train = x[train_ind]
        x_val = x[val_ind]
        y_train = y[train_ind]
        y_val = y[val_ind]
        with open(os.path.join(path_to_save,'val_ind.pkl'),'w') as f:
            pickle.dump(val_ind, f, pickle.HIGHEST_PROTOCOL)

        # y_train = to_categorical(y_train,2)
        # y_val = to_categorical(y_val,2)

        model = get_model(x_train.shape[1], x_train.shape[2],dropouts=dropouts)
        # model = EEGNet(2, Chans=19, Samples=150,
        #        dropoutRate=0.25, kernLength=75, F1=4,
        #        D=2, F2=8, norm_rate=0.25, dropoutType='Dropout')
        # tb = MyTensorBoard(log_dir=path_to_save)
        val_subjects = {subj:subjects[subj] for subj in subjects.keys() if subj != train_subject}
        val_subj_aucs = PerSubjAucMetricHistory(val_subjects)
        same_subj_auc = AucMetricHistory(save_best_by_auc=True, path_to_save=model_path)
        history = model.fit(x_train, y_train, epochs=150,validation_data=(x_val,y_val),
                            callbacks=[same_subj_auc], batch_size=64,shuffle=True)

        # multi_auc_loging(history.history, title='dropouts_%.2f_%.2f_%.2f' % dropouts,
        #                  val_subject_numbers=val_subjects.keys(),
        #                  path_to_save=path_to_save)
        single_auc_loging(history.history, hyperparam_name, path_to_save=path_to_save)


        with open('./res/simple_ebci/res.txt','a') as f:
            f.write('%d  %.3f  %d\n' %(train_subject,max(history.history['val_auc']),np.argmax(history.history['val_auc'])))
        print 'end'
###############################################################################################################