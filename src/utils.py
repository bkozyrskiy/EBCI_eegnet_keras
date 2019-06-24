import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import operator
from sklearn.model_selection import train_test_split
import numpy as np
import os
import random
import tensorflow as tf

# def loging(history,title):
#     fig = plt.figure()
#     plt.plot(history['loss'], figure=fig)
#     plt.plot(history['val_loss'], figure=fig)
#     plt.plot(history['val_auc'], figure=fig)
#     min_index, min_value = min(enumerate(history['val_loss']), key=operator.itemgetter(1))
#     plt.title('%s_min_%.3f_epoch%d' % (title, min_value, min_index))
#     plt.savefig('./res/%s.png' % (title), figure=fig)
#     with open('./res/%s.pkl'%title,'wb') as output:
#         pickle.dump(history,output,pickle.HIGHEST_PROTOCOL)

def set_seed(seed_value = 0):
    ''' Set detereministic seed'''
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.set_random_seed(seed_value)

def clean_bad_auc_models(model_path,history):
    '''
    Function clear all,but best (by auc) models in specified dir. Shoud be used with ModelCheckpoint and
    AucMetricHistory callbacks
    :param model_path: Dir, where placed all saved models, their names should be written in {epoch:02d}.hdf5 pattern
    :param history: dict, history of model's learning, which used to find best epoch by auc history
    :return:
    '''
    best_auc_history = np.argmax(history['val_auc']) + 1 #because ModelCheckpoint counts from 1
    best_checpoint = '%02d.hdf5' %best_auc_history
    for checkpoint_name in os.listdir(model_path):
        if checkpoint_name != best_checpoint:
            os.remove(os.path.join(model_path,checkpoint_name))


def single_auc_loging(history,title,path_to_save):
    """
    Function for ploting nn-classifier performance. It makes two subplots.
    First subplot with train and val losses
    Second with val auc
    Function saves plot as a picture and as a pkl file

    :param history: history field of history object, witch returned by model.fit()
    :param title: Title for picture (also used as filename)
    :param path_to_save: Path to save file
    :return:
    """
    f, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,12))

    if 'loss' in history.keys():
        loss_key = 'loss'  # for simple NN
    elif 'class_out_loss' in history.keys():
        loss_key = 'class_out_loss'  # for DAL NN
    else:
        raise ValueError('Not found correct key for loss information in history')

    ax1.plot(history[loss_key],label='cl train loss')
    ax1.plot(history['val_%s' %loss_key],label='cl val loss')
    ax1.legend()
    min_loss_index,max_loss_value = min(enumerate(history['val_loss']), key=operator.itemgetter(1))
    ax1.set_title('min_loss_%.3f_epoch%d' % (max_loss_value, min_loss_index))
    ax2.plot(history['val_auc'])
    max_auc_index, max_auc_value = max(enumerate(history['val_auc']), key=operator.itemgetter(1))
    ax2.set_title('max_auc_%.3f_epoch%d' % (max_auc_value, max_auc_index))
    f.suptitle('%s' % (title))
    plt.savefig('%s/%s.png' % (path_to_save,title), figure=f)
    plt.close()
    with open('%s/%s.pkl' % (path_to_save,title), 'wb') as output:
        pickle.dump(history,output,pickle.HIGHEST_PROTOCOL)

def multi_auc_loging(history,title,val_subject_numbers,path_to_save):
    """
    Function for ploting nn-classifier performance on different subjects. It makes N poctures,where N is number of
    different val sets (i.e. subjects) Each picture has two subplots.
    First subplot with train and val losses
    Second with val auc
    Function saves plots as a pictures and as single pkl file

    :param history: history field of history object, witch returned by model.fit()
    :param title: Title for picture (also used as filename)
    :param subject_numbers: list of subject's numbers, used for val sets
    :param path_to_save: Path to save file
    :return:
    """
    aucs = {}
    for subj in val_subject_numbers:
        f, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,12))
        if 'loss' in history.keys():
            loss_key = 'loss'                       #for simple NN
        elif 'class_out_loss' in history.keys():
            loss_key = 'class_out_loss'             #for DAL NN
        else:
            raise ValueError('Not found correct key for loss information in history')
        ax1.plot(history[loss_key],label='train loss')
        ax1.plot(history['val_loss_%d' %subj],label='val loss')
        ax1.legend()
        ax2.plot(history['val_auc_%d' %subj])
        min_loss_index, min_loss_value = min(enumerate(history['val_loss_%d' %subj]), key=operator.itemgetter(1))
        max_auc_index, max_auc_value = max(enumerate(history['val_auc_%d' %subj]), key=operator.itemgetter(1))
        aucs[subj] =  max_auc_value
        plt.title('min_val_loss:%.3f_epoch%d;_max_auc:%.3f_epoch:%d' % (min_loss_value, min_loss_index,
                                                                        max_auc_value, max_auc_index),loc='right')
        plt.savefig('%s/%d.png' % (path_to_save, subj), figure=f)
        plt.close()
    with open('%s/%s.pkl' %(path_to_save,title), 'wb') as output:
        pickle.dump(history,output,pickle.HIGHEST_PROTOCOL)

    with open('%s/res.txt' %(path_to_save),'wb') as f:
        mean_auc = reduce(lambda x,y: x+y,aucs.values())/float(len(aucs))
        for subj in sorted(aucs.keys()):
            f.write('%d %f\n' %(subj,aucs[subj]))
        f.write('Mean AUC %f\n' %mean_auc)

def multisubj_val_split(subjects,val_split=0.2,random_state=42):
    """
    This function make joins data from input subjects and splits them on train and test sets
    :param subjects: dict {subj_number:(x,y)}
    :return: x_train, x_val, y_train, y_val
    """
    tmp = []
    for subj in subjects.keys():
        tmp.append(train_test_split(subjects[subj][0], subjects[subj][1], test_size=val_split,
                                    stratify=subjects[subj][1], random_state=random_state))
    tmp = zip(*tmp)

    x_train = np.concatenate(tmp[0], axis=0)
    x_val = np.concatenate(tmp[1], axis=0)
    y_train = np.concatenate(tmp[2])
    y_val = np.concatenate(tmp[3])
    #save domain info about validation data
    return x_train, x_val, y_train, y_val




