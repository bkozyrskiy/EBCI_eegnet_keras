from keras.callbacks import Callback
from keras import backend as K
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score
import os
import matplotlib.pyplot as plt
import shutil
import numpy as np

class PerSubjAucMetricHistory(Callback):
    """
    This callback for testing model on each subject separately during training. It writes auc for every subject to the
    history object
    """
    def __init__(self,subjects):
        self.subjects = subjects
        super(PerSubjAucMetricHistory,self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        for subj in self.subjects.keys():
            x,y = self.subjects[subj]

            y_pred = self.model.predict(x, verbose=0)
            if isinstance(y_pred,list):
                y_pred = y_pred[0]
            if len(y_pred.shape) == 1:
                y_pred = to_categorical(y_pred,2)
            if len(y.shape) == 1:
                y = to_categorical(y,2)
            y_pred = to_categorical(y_pred) if (len(y_pred.shape)==1) else y_pred
            logs['val_auc_%s' %(subj)] = roc_auc_score(y[:,1], y_pred[:,1])
            if type(self.model.output) == list:
                fake_subj_labels = np.zeros((len(y),self.model.output[1].shape._dims[1]._value))
                logs['val_loss_%s' % (subj)] = self.model.evaluate(x,[y,fake_subj_labels], verbose=0)[0]
            else:
                logs['val_loss_%s' % (subj)] = self.model.evaluate(x,y, verbose=0)[0]

class AucMetricHistory(Callback):
    def __init__(self,save_best_by_auc=False,path_to_save=None):
        super(AucMetricHistory, self).__init__()
        self.save_best_by_auc=save_best_by_auc
        self.path_to_save = path_to_save
        self.best_auc = 0
        self.best_epoch = 1
        if save_best_by_auc and (path_to_save is None):
            raise ValueError('Specify path to save the model')

    def on_epoch_end(self, epoch, logs={}):
        x_val,y_val = self.validation_data[0],self.validation_data[1]
        y_pred = self.model.predict(x_val,batch_size=len(y_val), verbose=0)
        if isinstance(y_pred,list):
            y_pred = y_pred[0]
        current_auc = roc_auc_score(y_val, y_pred)
        logs['val_auc'] = current_auc

        if current_auc > self.best_auc:
            if self.save_best_by_auc:
                prev_model_path = os.path.join(self.path_to_save,'best_on_auc_%d_%.2f.hdf5' %(self.best_epoch,self.best_auc))
                if os.path.isfile(prev_model_path):
                    os.remove(prev_model_path)

                path_to_file = os.path.join(self.path_to_save, 'best_on_auc_%d_%.2f.hdf5' % (epoch,current_auc))
                self.model.save(path_to_file)

            self.best_auc = current_auc
            self.best_epoch = epoch

class DomainActivations(Callback):
    def __init__(self, x_train,y_train, subj_label_train,path_to_save):
        super(DomainActivations, self).__init__()
        self.path_to_save = '%s/domain_activations_grl/' % path_to_save
        self.x_train = x_train
        self.y_train = y_train
        self.subj_label_train = subj_label_train
        plt.plot(subj_label_train.argmax(axis=1))
        if os.path.isdir(self.path_to_save):
            shutil.rmtree(self.path_to_save)
        os.makedirs(self.path_to_save)
        plt.savefig(os.path.join('%s/class_distr' % self.path_to_save))
        plt.close()
    def _log_domain_activations(self, domain_label_pred, domain_label,pic_name):
        activations = (domain_label_pred * domain_label).sum(axis=1)
        plt.plot(activations)
        # plt.plot(activations[self.y_train[:,1] == 1])
        plt.savefig(os.path.join('%s/%s' % (self.path_to_save, pic_name)))
        plt.close()

    def on_epoch_end(self, epoch, logs=None):
        if epoch %10 ==0:
            self._log_domain_activations(self.model.predict(self.x_train, verbose=0)[1],self.subj_label_train,'%d_train' % epoch)

class MyTensorBoard(Callback):
    """Tensorboard basic visualizations.
    [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.
    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```sh
    tensorboard --logdir=/full_path_to_your_logs
    ```
    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(MyTensorBoard, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        global tf, projector
        import tensorflow as tf
        from tensorflow.contrib.tensorboard.plugins import projector
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
        self.batch_size = batch_size

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    tf.summary.histogram(mapped_weight_name, weight)
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'
                        grads = [
                            grad.values if is_indexed_slices(grad) else grad
                            for grad in grads]
                        tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = K.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       shape[1],
                                                       1])
                        elif len(shape) == 3:  # convnet case
                            if K.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0],
                                                       shape[1],
                                                       shape[2],
                                                       1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       1,
                                                       1])
                        else:
                            # not possible to handle 3D convnets etc.
                            continue

                        shape = K.int_shape(w_img)
                        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                        tf.summary.image(mapped_weight_name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.embeddings_freq:
            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']

            embeddings = {layer.name: layer.weights[0]
                          for layer in self.model.layers
                          if layer.name in embeddings_layer_names}

            self.saver = tf.train.Saver(list(embeddings.values()))

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings.keys()}

            config = projector.ProjectorConfig()
            self.embeddings_ckpt_path = os.path.join(self.log_dir,
                                                     'keras_embedding.ckpt')

            for layer_name, tensor in embeddings.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if not self.validation_data and self.histogram_freq:
            raise ValueError('If printing histograms, validation_data must be '
                             'provided, and cannot be a generator.')
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] for x in val_data]
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                self.saver.save(self.sess,
                                self.embeddings_ckpt_path,
                                epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value =  K.get_value(self.model.optimizer.lr)
        summary_value.tag = 'learning_rate'
        self.writer.add_summary(summary, epoch)
        summary = tf.Summary()
        summary_value = summary.value.add()
        x_val, y_val = self.validation_data[0], self.validation_data[1]

        y_pred = self.model.predict(x_val, verbose=0)
        if isinstance(y_pred,list):
            y_pred = y_pred[0]
        summary_value.simple_value = (roc_auc_score(y_val[:, 1], y_pred[:, 1]))
        summary_value.tag = 'val_auc'
        self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()

