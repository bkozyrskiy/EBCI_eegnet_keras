from keras.models import Model
from keras.layers import Input, MaxPooling2D, Conv1D,Conv2D, BatchNormalization,Dropout,Flatten,Dense
from keras.layers.core import Reshape
from keras.regularizers import l1_l2
from keras.optimizers import Adam
from keras.callbacks import Callback,ReduceLROnPlateau,EarlyStopping
from my_callbacks import MyTensorBoard
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from data_ern import *
from grl_tf import GradientReversal
from my_callbacks import  AucMetricHistory
from utils import single_auc_loging

def get_model(time_samples_num,channels_num,dropouts,hp_lambda=1):
    # rn_init = RandomNormal(stddev=0.001,seed=1)
    #First
    num_of_filt = 16

    input=Input(shape=(time_samples_num, channels_num, ))
    # convolved = Conv1D(num_of_filt,kernel_size=(1),activation='elu',padding='same',kernel_regularizer=l1_l2(0.0001))(input)
    convolved = Conv1D(num_of_filt, kernel_size=(1), activation='elu', padding='same')(input)
    convolved = Reshape((1, num_of_filt, time_samples_num))(convolved)
    #
    #
    b_normed = BatchNormalization(axis=2)(convolved)
    dropouted = Dropout(dropouts[0])(b_normed)
    #
    # #second
    num_of_filt=4
    convolved =Conv2D(num_of_filt, kernel_size=(2, 32),
            activation='elu',
           #kernel_regularizer=l1_l2(0.0000),
            data_format='channels_first',
           padding='same')(dropouted)
    b_normed = BatchNormalization(axis=1)(convolved)
    pooled = MaxPooling2D(pool_size=(2,4),data_format='channels_first')(b_normed)
    dropouted = Dropout(dropouts[1])(pooled)
    #
    # #Third
    num_of_filt = 4
    convolved = Conv2D(num_of_filt, kernel_size=(8,4),
                       activation='elu',
                       #kernel_regularizer=l1_l2(0.0000),
                       data_format='channels_first',
                       padding='same')(dropouted)
    b_normed = BatchNormalization(axis=1)(convolved)
    pooled = MaxPooling2D(pool_size=(2, 4),data_format='channels_first')(b_normed)
    dropouted = Dropout(dropouts[2],seed=1)(pooled)

    #Fourth
    flatten = Flatten()(dropouted)
    class_out = Dense(2,activation='softmax',name='class_out')(flatten)
    domain_branch = GradientReversal(hp_lambda=hp_lambda)(flatten)
    # domain_out = Dense(num_subjects,activation='softmax',name='domain_out')(domain_branch)
    domain_out = Dense(1, activation='sigmoid', name='domain_out')(domain_branch)
    classification_model = Model(inputs=input,outputs=[class_out,domain_out])
    opt = Adam(lr=0.005)
    classification_model.compile(optimizer=opt,loss={'class_out':'categorical_crossentropy',
                                      'domain_out' : 'binary_crossentropy'})

    # model.compile(optimizer=opt, loss='categorical_crossentropy')
    # metrics=['accuracy'])
    tsne_model = Model(inputs=input,outputs=flatten)
    return classification_model, tsne_model





if __name__ =='__main__':
    data = DataERN('/home/likan_blk/BCI/eegnet/data_ern')
    # x, y = data.get_data(shuffle=True, resample_to=128, subject_indices=range(12),balance_classes=True)
    # x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=0)

    x_train, y_train,subj_labels_train = data.get_data(shuffle=True, resample_to=128, subject_indices=range(2), balance_classes=True)
    x_val, y_val,subj_labels_val = data.get_data(shuffle=False, resample_to=128, subject_indices=range(2,3), balance_classes=True)
    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)
    subjects_num =max(subj_labels_train.max(), subj_labels_val.max()) + 1
    subj_labels_train = to_categorical(subj_labels_train,subjects_num)
    subj_labels_val = to_categorical(subj_labels_val, subjects_num)
    # x_train, x_val, y_train, y_val = train_test_split(x, np.hstack((y,subj_label)).transpose(),
    #                                                   test_size=0.2, random_state=0)
    #
    # class_train_labels,subj_train_labels = y_train.transpose()[:,0],y_train.transpose()[:,1]
    # class_val_labels, subj_val_labels = y_val.transpose()[:, 0], y_val.transpose()[:, 1]

    tb = MyTensorBoard(log_dir='./logs/DAL_GRL_res/', write_graph=True, write_grads=True,histogram_freq=1)
    dropouts = [0.5, 0.5, 0.5]
    model = get_model(x_train.shape[1], x_train.shape[2], subjects_num, dropouts)
    #
    auc_log = AucMetricHistory()
    activations_log = DomainActivations(x_train,y_train,subj_labels_train)
    #
    num_epochs = 300
    history = model.fit(x_train, {'class_out' : y_train,'domain_out':subj_labels_train},
                        validation_data=[x_val, {'class_out' : y_val,'domain_out':subj_labels_val}],
                        epochs=num_epochs, callbacks=[tb,auc_log,activations_log], batch_size=50, shuffle=True)

    # log_domain_activations(model.predict(x_train)[1], subj_labels_train, '%d_train' % num_epochs)
    single_auc_loging(history.history, title='DAL_GRL_ern')
    print 'end'
