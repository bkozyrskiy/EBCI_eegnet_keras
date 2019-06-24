import keras.backend as K
from deepexplain.tensorflow import DeepExplain
from keras.models import Model,load_model
import matplotlib.pyplot as plt
import os
import pickle
import tensorflow as tf




def grads_wrt_input(model,x,y):
    '''
    Funtion returns gradient of output of last learnable layer (before softmax) w.r.t to input data

    :param model: keras model
    :param x: input samples
    :param y: lables of  input samples
    :return:
    '''
    outputTensor = model.layers[-2].output
    listOfVariableTensors = model.inputs[0]
    gradients = K.gradients(outputTensor*y, listOfVariableTensors)
    sess = K.get_session()
    evaluated_gradients = sess.run(gradients, feed_dict={model.input: x})[0]
    return evaluated_gradients


def apply_deep_expl(path_to_model,method,x,y):
    '''
    Funtion returns feature relevance using Deep Explain package
    :param path_to_model:
    :param method:
    :param x:
    :param y:
    :return:
    '''
    K.clear_session()
    with DeepExplain(session=K.get_session()) as de:
        model = load_model(path_to_model)
        net_predictions = model.predict(x)
        input_tensor = model.layers[0].input
        target_tensor = model.layers[-2].output
        # fModel = Model(inputs=input_tensor, outputs=model.layers[-2].output)
        # target_tensor = fModel(input_tensor)
        # print input_tensor.get_shape()
        # print target_tensor.get_shape()
        attributions = de.explain(method, target_tensor * y, input_tensor, x)
        # print type(attributions)
        # print type(net_predictions)
    return attributions,net_predictions


def simple_plot(attributions):
    plt.imshow(attributions.mean(axis=0))