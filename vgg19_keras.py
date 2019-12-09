import tensorflow as tf
#from keras.applications.vgg19 import preprocess_input
from ops import L1_mean_loss
import numpy as np

class VGGLoss(tf.keras.Model):
    def __init__(self):
        super(VGGLoss, self).__init__(name='VGGLoss')
        self.vgg = Vgg19()
        self.layer_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def call(self, x, y):
        x = ((x + 1) / 2) * 255.0
        y = ((y + 1) / 2) * 255.0
        x_vgg, y_vgg = self.vgg(preprocess_input(x)), self.vgg(preprocess_input(y))

        loss = 0

        for i in range(len(x_vgg)):
            #y_vgg_detach = tf.stop_gradient(y_vgg[i])
            y_vgg_detach = y_vgg[i]
            loss += self.layer_weights[i] * L1_mean_loss(x_vgg[i], y_vgg_detach)

        return loss

class Vgg19(tf.keras.Model):
    def __init__(self, trainable=False):
        super(Vgg19, self).__init__(name='Vgg19')
        vgg_pretrained_features = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False)

        if trainable is False:
            vgg_pretrained_features.trainable = False

        vgg_pretrained_features = vgg_pretrained_features.layers

        self.slice1 = tf.keras.Sequential()
        self.slice2 = tf.keras.Sequential()
        self.slice3 = tf.keras.Sequential()
        self.slice4 = tf.keras.Sequential()
        self.slice5 = tf.keras.Sequential()

        for x in range(1, 2):
            self.slice1.add(vgg_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add(vgg_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add(vgg_pretrained_features[x])
        for x in range(8, 13):
            self.slice4.add(vgg_pretrained_features[x])
        for x in range(13, 18):
            self.slice5.add(vgg_pretrained_features[x])

    def call(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

def preprocess_input(x, data_format=None, mode='caffe', **kwargs):
    #backend, _, _, _ = get_submodules_from_kwargs(kwargs)
    backend = tf.keras.backend

    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    if isinstance(x, np.ndarray):
        return _preprocess_numpy_input(x, data_format=data_format,
                                       mode=mode, **kwargs)
    else:
        return _preprocess_symbolic_input(x, data_format=data_format,
                                          mode=mode, **kwargs)

def _preprocess_numpy_input(x, data_format, mode, **kwargs):
    #backend, _, _, _ = get_submodules_from_kwargs(kwargs)
    backend = tf.keras.backend

    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(backend.floatx(), copy=False)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x

    if mode == 'torch':
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x

def _preprocess_symbolic_input(x, data_format, mode, **kwargs):
    #backend, _, _, _ = get_submodules_from_kwargs(kwargs)
    backend = tf.keras.backend

    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x

    if mode == 'torch':
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if backend.ndim(x) == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

    #mean_tensor = backend.constant(-np.array(mean))
    mean_tensor = tf.convert_to_tensor(-np.array(mean))
    if std:
        std_tensor = tf.convert_to_tensor(np.array(1.0/std))
    else:
        std_tensor = None

    # Zero-center by mean pixel
    #if backend.dtype(x) != backend.dtype(mean_tensor):
    #    x = backend.bias_add(
    #        x, backend.cast(mean_tensor, backend.dtype(x)),
    #        data_format=data_format)
    #else:
    #    x = backend.bias_add(x, mean_tensor, data_format)
    x = bias_add(x, tf.cast(mean_tensor, dtype=x.dtype), data_format)
    if std_tensor is not None:
        x = scale_multiply(x, tf.cast(std_tensor, dtype=x.dtype), data_format)
    return x

def bias_add(x, bias, data_format=None):
    bias_shape = bias.shape
    if data_format == 'channels_first':
      x = x + tf.reshape(bias, (1, bias_shape[0], 1, 1))
    elif data_format == 'channels_last':
      x = x + tf.reshape(bias, (1, 1, 1, bias_shape[0]))
    return x 

def scale_multiply(x, scale, data_format=None):
    scale_shape = scale.shape
    if data_format == 'channels_first':
      x = x + tf.reshape(scale, (1, scale_shape[0], 1, 1))
    elif data_format == 'channels_last':
      x = x + tf.reshape(scale, (1, 1, 1, scale_shape[0]))
    return x
