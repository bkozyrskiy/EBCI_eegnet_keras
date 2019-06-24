from keras import Model
from keras.layers import Input, MaxPooling2D, AveragePooling2D, Conv1D, Conv2D, BatchNormalization, Dropout, Flatten, Dense, Activation
from keras.layers.core import Reshape, Permute
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.regularizers import l1_l2
from keras.optimizers import Adam
import keras.backend as K
from keras.constraints import max_norm
from keras.layers import SpatialDropout2D
from keras.utils.generic_utils import get_custom_objects

def my_EEGnet(time_samples_num, channels_num,params):
    # rn_init = RandomNormal(stddev=0.001,seed=1)
    # First
    num_of_filt = 16

    input = Input(shape=(time_samples_num, channels_num,))
    convolved = Conv1D(num_of_filt, kernel_size=(1), activation='elu', padding='same',
                       kernel_regularizer=l1_l2(0.0001))(input)
    convolved = Reshape((1, num_of_filt, time_samples_num))(convolved)
    #
    #
    b_normed = BatchNormalization(axis=2)(convolved)
    dropouted = Dropout(params['dropouts0'])(b_normed)
    #
    # #second
    num_of_filt = 4
    convolved = Conv2D(num_of_filt, kernel_size=(2, 32),
                       activation='elu',
                       # kernel_regularizer=l1_l2(0.0000),
                       data_format='channels_first',
                       padding='same')(dropouted)
    b_normed = BatchNormalization(axis=1)(convolved)
    pooled = MaxPooling2D(pool_size=(2, 4), data_format='channels_first')(b_normed)
    dropouted = Dropout(params['dropouts1'])(pooled)
    #
    # #Third
    num_of_filt = 4
    convolved = Conv2D(num_of_filt, kernel_size=(8, 4),
                       activation='elu',
                       # kernel_regularizer=l1_l2(0.0000),
                       data_format='channels_first',
                       padding='same')(dropouted)
    b_normed = BatchNormalization(axis=1)(convolved)
    pooled = MaxPooling2D(pool_size=(2, 4), data_format='channels_first')(
        b_normed)  # 41 time sample point affects this feature
    dropouted = Dropout(params['dropouts2'], seed=1)(pooled)

    # Fourth
    flatten = Flatten()(dropouted)
    out = Dense(2, activation=None)(flatten)
    out = Activation(activation='softmax')(out)
    classification_model = Model(inputs=input, outputs=out)
    opt = Adam(lr=0.005)
    classification_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return classification_model

def EEGNet_old(params,Chans=19, Samples=128, kernels=[(2, 32), (8, 4)], strides=(2, 4)):
    """ Keras Implementation of EEGNet_v1 (https://arxiv.org/abs/1611.08024v2)
    This model is the original EEGNet model proposed on arxiv
            https://arxiv.org/abs/1611.08024v2

    with a few modifications: we use striding instead of max-pooling as this
    helped slightly in classification performance while also providing a
    computational speed-up.

    Note that we no longer recommend the use of this architecture, as the new
    version of EEGNet performs much better overall and has nicer properties.

    Inputs:

        nb_classes     : total number of final categories
        Chans, Samples : number of EEG channels and samples, respectively
        regRate        : regularization rate for L1 and L2 regularizations
        dropoutRate    : dropout fraction
        kernels        : the 2nd and 3rd layer kernel dimensions (default is
                         the [2, 32] x [8, 4] configuration)
        strides        : the stride size (note that this replaces the max-pool
                         used in the original paper)

    """

    # start the model
    input_main = Input((1, Chans, Samples))
    layer1 = Conv2D(16, (Chans, 1), input_shape=(1, Chans, Samples),
                    kernel_regularizer=l1_l2(l1=params['regRate'], l2=params['regRate']),data_format='channels_first')(input_main)
    layer1 = BatchNormalization(axis=1)(layer1)
    layer1 = Activation('elu')(layer1)
    layer1 = Dropout(params['dropoutRate1'])(layer1)

    permute_dims = 2, 1, 3
    permute1 = Permute(permute_dims)(layer1)

    layer2 = Conv2D(4, kernels[0], padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=params['regRate']),
                    strides=strides,data_format='channels_first')(permute1)
    layer2 = BatchNormalization(axis=1)(layer2)
    layer2 = Activation('elu')(layer2)
    layer2 = Dropout(params['dropoutRate2'])(layer2)

    layer3 = Conv2D(4, kernels[1], padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=params['regRate']),
                    strides=strides,data_format='channels_first')(layer2)
    layer3 = BatchNormalization(axis=1)(layer3)
    layer3 = Activation('elu')(layer3)
    layer3 = Dropout(params['dropoutRate3'])(layer3)

    flatten = Flatten(name='flatten')(layer3)

    dense = Dense(2, name='dense')(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    model = Model(inputs=input_main, outputs=softmax)
    opt = Adam(lr=params['lr'])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))

def ShallowConvNet(params,Chans=64, Samples=128):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.

    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25

    Note that this implementation has not been verified by the original
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations.
    """

    get_custom_objects().update({'square': Activation(square)})
    get_custom_objects().update({'log': Activation(log)})
    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(40, (1, 7),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(3., axis=(0, 1, 2)),
                    data_format='channels_first')(input_main)
    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(3., axis=(0, 1, 2)),
                    data_format='channels_first')(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 18), strides=(1, 4), data_format='channels_first')(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(params['dropoutRate'])(block1)
    flatten = Flatten()(block1)
    dense = Dense(2, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    model = Model(inputs=input_main, outputs=softmax)
    opt = Adam(lr=params['lr'])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def EEGNet(params, nb_classes, Chans=64, Samples=128, kernLength=64,
           F2=8, dropoutType='Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-4,2 model as discussed
    in the paper. This model should do pretty well in general, although as the
    paper discussed the EEGNet-8,2 (with 8 temporal kernels and 2 spatial
    filters per temporal kernel) can do slightly better on the SMR dataset.
    Other variations that we found to work well are EEGNet-4,1 and EEGNet-8,1.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.

    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 4, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(1, Chans, Samples))

    ##################################################################
    block1 = Conv2D(params['F1'], (1, kernLength), padding='same',
                    input_shape=(1,Chans, Samples),
                    use_bias=False)(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=params['D'],
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(params['dropoutRate1'])(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(params['dropoutRate2'])(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(params['norm_rate']))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    model = Model(inputs=input1, outputs=softmax)
    opt = Adam(lr=params['lr'])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model