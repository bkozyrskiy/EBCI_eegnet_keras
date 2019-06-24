from keras import Model
from keras.layers import Input, MaxPooling2D, Conv1D,Conv2D, BatchNormalization,Dropout,Flatten,Dense,Activation
from keras.layers.core import Reshape
from keras.regularizers import l1_l2
from keras.optimizers import Adam
import  keras.backend as K
import tensorflow as tf


def auc_metric(y_true,y_pred):
    score, up_opt = tf.metrics.auc(y_true,y_pred,num_thresholds=54)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

def get_model(time_samples_num,channels_num,dropouts):
    # rn_init = RandomNormal(stddev=0.001,seed=1)
    #First
    num_of_filt = 16

    input=Input(shape=(time_samples_num, channels_num, ))
    convolved = Conv1D(num_of_filt,kernel_size=(1),activation='elu',padding='same',kernel_regularizer=l1_l2(0.0001))(input)
    convolved = Reshape((1, num_of_filt, time_samples_num))(convolved)
    #
    #
    b_normed = BatchNormalization(axis=2)(convolved)
    dropouted = Dropout(dropouts[0])(b_normed)
    #
    # #second
    num_of_filt = 4
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
    pooled = MaxPooling2D(pool_size=(2, 4),data_format='channels_first')(b_normed) # 41 time sample point affects this feature
    dropouted = Dropout(dropouts[2],seed=1)(pooled)

    #Fourth
    flatten = Flatten()(dropouted)
    out = Dense(2,activation=None)(flatten)
    out = Activation(activation='softmax')(out)
    classification_model = Model(inputs=input,outputs=out)
    opt = Adam(lr=0.0009)
    classification_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

    return classification_model





