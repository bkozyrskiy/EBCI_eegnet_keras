from src.NN import *
from src.data_ern import *
from src.my_callbacks import PerSubjAucMetricHistory
import itertools
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import shutil
from src.utils import multi_auc_loging


#Standart eegnet test on ERN data set. Simple NN, trained on [1:k] subjects and tested on [k+1:n] subjects
################################################################################################################
# if __name__=='__main__':
#     data = DataERN('/home/likan_blk/BCI/eegnet/data_ern')
#     # x, y = data.get_data(shuffle=True, resample_to=128, subject_indices=range(12),balance_classes=True)
#     # x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=0)
#     x_train, y_train,_ = data.get_data(shuffle=True, resample_to=128, subject_indices=range(2), balance_classes=True)
#     x_val, y_val,_ = data.get_data(shuffle=True, resample_to=128, subject_indices=range(2,3), balance_classes=True)
#     y_train = to_categorical(y_train,2)
#     y_val = to_categorical(y_val,2)
#     # x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
#     # x_val,y_val = data.get_data(shuffle=False, resample_to=128, subject_indices=range(12,16))
# #
#     model = get_model(x_train.shape[1], x_train.shape[2],dropouts=(0.5,0.5,0.5))
#     tb = MyTensorBoard(log_dir='./logs')
#     # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=32, epsilon=0.01, min_lr=0.00001)
#     log = LossMetricHistory()
#     history = model.fit(x_train, y_train, epochs=200, validation_data=[x_val,y_val], callbacks=[tb,log], batch_size=64, shuffle=True)
#     loging(history.history, title='dropouts_%.2f_%.2f_%.2f' % (0.5,0.5,0.5))
#     print 'end'
################################################################################################################
#Test, where two subjects used for training and all others for validating SEPARATELY. Simpler netwrok, data ern
###############################################################################################################
def get_data_by_subj(x,y,subj_labels):
    return {subj:(x[np.where(subj_labels==subj)],y[np.where(subj_labels==subj)]) for subj in np.unique(subj_labels)}

if __name__=='__main__':
    all_subjects_indeces = range(16)
    data = DataERN('/home/likan_blk/BCI/eegnet/data_ern')
    x, y, subject_labels = data.get_data(shuffle=False, resample_to=128, subject_indices=all_subjects_indeces,
                                         balance_classes=True)
    subjects = get_data_by_subj(x, y, subject_labels)
    subjects_sets = [(12,13),(13,16),(16,26)]
    for train_subjects in subjects_sets:
        path_to_save = './res/simple_ERN/%s_%s/' %(train_subjects[0],train_subjects[1])
        if os.path.isdir(path_to_save):
            shutil.rmtree(path_to_save)
        os.makedirs(path_to_save)
        # checkpoint = ModelCheckpoint(path_to_save, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        x_train = np.concatenate([subjects[subj][0] for subj in  train_subjects])
        y_train = np.concatenate([subjects[subj][1] for subj in train_subjects])
        y_train = to_categorical(y_train,2)
        model = get_model(x.shape[1], x.shape[2],dropouts=(0.25,0.25,0.25))
        # tb = MyTensorBoard(log_dir=path_to_save)
        val_subjects = {subj:subjects[subj] for subj in subjects.keys() if subj not in train_subjects}
        subj_aucs = PerSubjAucMetricHistory(val_subjects)
        history = model.fit(x_train, y_train, epochs=200, callbacks=[subj_aucs], batch_size=64, shuffle=True)
        multi_auc_loging(history.history, title='dropouts_%.2f_%.2f_%.2f' % (0.25,0.25,0.25),
                         val_subject_numbers=val_subjects.keys(),
                         path_to_save=path_to_save)
        print 'end'
###############################################################################################################