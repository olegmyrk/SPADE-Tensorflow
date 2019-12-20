import math
import numpy as np
import tensorflow as tf
#import tensorflow_addons as tf_addons
from utils import pytorch_xavier_weight_factor, pytorch_kaiming_weight_factor

##################################################################################
# Initialization
##################################################################################

"""
pytorch xavier (gain)
https://pytorch.org/docs/stable/_modules/torch/nn/init.html

if uniform :
    factor = gain * gain
    mode = 'FAN_AVG'
else :
    SPADE use, gain=0.02
    factor = (gain * gain) / 1.3
    mode = 'FAN_AVG'
    
pytorch : trunc_stddev = gain * sqrt(2 / (fan_in + fan_out))
tensorflow  : trunc_stddev = sqrt(1.3 * factor * 2 / (fan_in + fan_out))

"""

"""
pytorch kaiming (a=0)
https://pytorch.org/docs/stable/_modules/torch/nn/init.html

if uniform :
    a = 0 -> gain = sqrt(2)
    factor = gain * gain
    mode='FAN_IN'
else :
    a = 0 -> gain = sqrt(2)
    factor = (gain * gain) / 1.3
    mode = 'FAN_OUT', but SPADE use 'FAN_IN'
    
pytorch : trunc_stddev = gain * sqrt(2 / (fan_in + fan_out))
tensorflow  : trunc_stddev = sqrt(1.3 * factor * 2 / (fan_in + fan_out))

"""

factor, mode, uniform = pytorch_xavier_weight_factor(gain=0.02, uniform=False)
#weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
weight_init = tf.compat.v1.variance_scaling_initializer(scale=factor, mode=mode.lower(), distribution='uniform' if uniform else 'truncated_normal')
# tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

weight_regularizer = None
weight_regularizer_fully = None

# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# Truncated_normal : tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
# Orthogonal : tf.orthogonal_initializer(0.02)

##################################################################################
# Regularization
##################################################################################

# l2_decay : tf.contrib.layers.l2_regularizer(0.0001)
# orthogonal_regularizer : orthogonal_regularizer(0.0001) # orthogonal_regularizer_fully(0.0001)


##################################################################################
# Layer
##################################################################################

def get_trainable_variable(name, shape=None, dtype=None, initializer=None, regularizer=None):
  variable = tf.compat.v1.get_variable(name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer)
  tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, variable)
  return variable

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(tensor=x, paddings=[[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(tensor=x, paddings=[[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = get_trainable_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filters=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = get_trainable_variable("bias", [channels], initializer=tf.compat.v1.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.compat.v1.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def partial_conv(x, channels, kernel=3, stride=2, use_bias=True, padding='SAME', sn=False, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        if padding.lower() == 'SAME'.lower():
            with tf.compat.v1.variable_scope('mask', reuse=reuse):
                _, h, w, _ = x.get_shape().as_list()

                slide_window = kernel * kernel
                mask = tf.ones(shape=[1, h, w, 1])

                update_mask = tf.compat.v1.layers.conv2d(mask, filters=1,
                                               kernel_size=kernel, kernel_initializer=tf.compat.v1.constant_initializer(1.0),
                                               strides=stride, padding=padding, use_bias=False, trainable=False)

                mask_ratio = slide_window / (update_mask + 1e-8)
                update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
                mask_ratio = mask_ratio * update_mask

            with tf.compat.v1.variable_scope('x', reuse=reuse):
                if sn:
                    w = get_trainable_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                                        initializer=weight_init, regularizer=weight_regularizer)
                    x = tf.nn.conv2d(input=x, filters=spectral_norm(w), strides=[1, stride, stride, 1], padding=padding)
                else:
                    x = tf.compat.v1.layers.conv2d(x, filters=channels,
                                         kernel_size=kernel, kernel_initializer=weight_init,
                                         kernel_regularizer=weight_regularizer,
                                         strides=stride, padding=padding, use_bias=False)
                x = x * mask_ratio

                if use_bias:
                    bias = get_trainable_variable("bias", [channels], initializer=tf.compat.v1.constant_initializer(0.0))

                    x = tf.nn.bias_add(x, bias)
                    x = x * update_mask
        else:
            if sn:
                w = get_trainable_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                                    initializer=weight_init, regularizer=weight_regularizer)
                x = tf.nn.conv2d(input=x, filters=spectral_norm(w), strides=[1, stride, stride, 1], padding=padding)
                if use_bias:
                    bias = get_trainable_variable("bias", [channels], initializer=tf.compat.v1.constant_initializer(0.0))

                    x = tf.nn.bias_add(x, bias)
            else:
                x = tf.compat.v1.layers.conv2d(x, filters=channels,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=weight_regularizer,
                                     strides=stride, padding=padding, use_bias=use_bias)

        return x

def fully_connected(x, units, use_bias=True, sn=False, reuse=tf.compat.v1.AUTO_REUSE, scope='linear'):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = get_trainable_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer_fully)
            if use_bias:
                bias = get_trainable_variable("bias", [units],
                                       initializer=tf.compat.v1.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.compat.v1.layers.dense(x, units=units, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer_fully,
                                use_bias=use_bias)

        return x

def flatten(x):
    return tf.compat.v1.layers.flatten(x)


##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, sn=False, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    channel_in = x_init.get_shape().as_list()[-1]
    channel_middle = min(channel_in, channels)

    with tf.compat.v1.variable_scope(scope, reuse=reuse) :
        x = conv(x, channels=channel_middle, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_1')

        x = lrelu(x, 0.2)
        x = conv(x, channels=channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_2')

        if channel_in != channels :
            x_init = conv(x_init, channels=channels, kernel=1, stride=1, use_bias=False, sn=sn, scope='conv_shortcut')

        return x + x_init

def constin_resblock(x_init, channels, use_bias=True, sn=False, norm=True, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    channel_in = x_init.get_shape().as_list()[-1]
    channel_middle = min(channel_in, channels)

    with tf.compat.v1.variable_scope(scope, reuse=reuse) :
        x = constin(x_init, channel_in, use_bias=use_bias, sn=False, norm=norm, scope='norm_1')
        x = lrelu(x, 0.2)
        x = conv(x, channels=channel_middle, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_1')

        x = constin(x, channels=channel_middle, use_bias=use_bias, sn=False, norm=norm, scope='norm_2')
        x = lrelu(x, 0.2)
        x = conv(x, channels=channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_2')

        if channel_in != channels :
            x_init = constin(x_init, channels=channel_in, use_bias=use_bias, sn=False, norm=norm, scope='norm_shortcut')
            x_init = conv(x_init, channels=channels, kernel=1, stride=1, use_bias=False, sn=sn, scope='conv_shortcut')

        return x + x_init

def constin(x_init, channels, use_bias=True, sn=False, norm=True, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse) :
        if norm:
            #x = param_free_norm(x_init)
            x = batch_norm(x_init, scope="batch_norm")
        else:
            x = x_init

        _, x_h, x_w, x_c = x_init.get_shape().as_list()

        gamma = get_trainable_variable("gamma", shape=[x_c], initializer=weight_init, regularizer=weight_regularizer)
        beta = get_trainable_variable("beta", shape=[x_c], initializer=weight_init, regularizer=weight_regularizer)

        x = x * (1 + gamma) + beta

        return x

def constin_fcblock(x_init, channels, use_bias=True, sn=False, norm=True, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    channel_in = x_init.get_shape().as_list()[-1]
    channel_middle = min(channel_in, channels)

    with tf.compat.v1.variable_scope(scope, reuse=reuse) :
        x = constin_vector(x_init, channel_in, use_bias=use_bias, sn=False, norm=norm, scope='norm_1')
        x = lrelu(x, 0.2)
        x = fully_connected(x, units=channel_middle, use_bias=use_bias, sn=sn, scope='linear_1')

        x = constin_vector(x, channels=channel_middle, use_bias=use_bias, sn=False, norm=norm, scope='norm_2')
        x = lrelu(x, 0.2)
        x = fully_connected(x, units=channels, use_bias=use_bias, sn=sn, scope='linear_2')

        if channel_in != channels :
            x_init = constin_vector(x_init, channels=channel_in, use_bias=use_bias, sn=False, norm=norm, scope='norm_shortcut')
            x_init = fully_connected(x_init, units=channels, use_bias=False, sn=sn, scope='linear_shortcut')

        return x + x_init

def constin_vector(x_init, channels, use_bias=True, sn=False, norm=True, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse) :
        if norm:
            #x = param_free_norm(x_init)
            x = batch_norm(x_init, scope="batch_norm")
        else:
            x = x_init

        _, x_c = x_init.get_shape().as_list()

        gamma = get_trainable_variable("gamma", shape=[x_c], initializer=weight_init, regularizer=weight_regularizer)
        beta = get_trainable_variable("beta", shape=[x_c], initializer=weight_init, regularizer=weight_regularizer)

        x = x * (1 + gamma) + beta

        return x

def adain_resblock(context, x_init, channels, use_bias=True, sn=False, norm=True, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    channel_in = x_init.get_shape().as_list()[-1]
    channel_middle = min(channel_in, channels)

    with tf.compat.v1.variable_scope(scope, reuse=reuse) :
        x = adain(context, x_init, channel_in, use_bias=use_bias, sn=False, norm=norm, scope='norm_1')
        x = lrelu(x, 0.2)
        x = conv(x, channels=channel_middle, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_1')

        x = adain(context, x, channels=channel_middle, use_bias=use_bias, sn=False, norm=norm, scope='norm_2')
        x = lrelu(x, 0.2)
        x = conv(x, channels=channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_2')

        if channel_in != channels :
            x_init = adain(context, x_init, channels=channel_in, use_bias=use_bias, sn=False, norm=norm, scope='norm_shortcut')
            x_init = conv(x_init, channels=channels, kernel=1, stride=1, use_bias=False, sn=sn, scope='conv_shortcut')

        return x + x_init

def adain(context, x_init, channels, use_bias=True, sn=False, norm=True, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse) :
        if norm:
            #x = param_free_norm(x_init)
            x = batch_norm(x_init, scope="batch_norm")
        else:
            x = x_init

        _, x_h, x_w, x_c = x_init.get_shape().as_list()

        context_gamma = fully_connected(context, units=channels, use_bias=use_bias, sn=sn, scope='linear_gamma')
        context_beta = fully_connected(context, units=channels, use_bias=use_bias, sn=sn, scope='linear_beta')

        context_shape = [-1, 1, 1, channels]
        context_gamma = tf.reshape(context_gamma, context_shape)
        context_beta = tf.reshape(context_beta, context_shape)

        x = x * (1 + context_gamma) + context_beta

        return x

def adain_fcblock(context, x_init, channels, use_bias=True, sn=False, norm=True, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    channel_in = x_init.get_shape().as_list()[-1]
    channel_middle = min(channel_in, channels)

    with tf.compat.v1.variable_scope(scope, reuse=reuse) :
        x = adain_vector(context, x_init, channel_in, use_bias=use_bias, sn=False, norm=norm, scope='norm_1')
        x = lrelu(x, 0.2)
        x = fully_connected(x, units=channel_middle, use_bias=use_bias, sn=sn, scope='linear_1')

        x = adain_vector(context, x, channels=channel_middle, use_bias=use_bias, sn=False, norm=norm, scope='norm_2')
        x = lrelu(x, 0.2)
        x = fully_connected(x, units=channels, use_bias=use_bias, sn=sn, scope='linear_2')

        if channel_in != channels :
            x_init = adain_vector(context, x_init, channels=channel_in, use_bias=use_bias, sn=False, norm=norm, scope='norm_shortcut')
            x_init = fully_connected(x_init, units=channels, use_bias=False, sn=sn, scope='linear_shortcut')

        return x + x_init

def adain_vector(context, x_init, channels, use_bias=True, sn=False, norm=True, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse) :
        if norm:
            #x = param_free_norm(x_init)
            x = batch_norm(x_init, scope="batch_norm")
        else:
            x = x_init

        _, x_c = x_init.get_shape().as_list()

        context_gamma = fully_connected(context, units=channels, use_bias=use_bias, sn=sn, scope='linear_gamma')
        context_beta = fully_connected(context, units=channels, use_bias=use_bias, sn=sn, scope='linear_beta')

        x = x * (1 + context_gamma) + context_beta

        return x

def spade_resblock(segmap, x_init, channels, use_bias=True, sn=False, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    channel_in = x_init.get_shape().as_list()[-1]
    channel_middle = min(channel_in, channels)

    with tf.compat.v1.variable_scope(scope, reuse=reuse) :
        x = spade(segmap, x_init, channel_in, use_bias=use_bias, sn=False, scope='norm_1')
        x = lrelu(x, 0.2)
        x = conv(x, channels=channel_middle, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_1')

        x = spade(segmap, x, channels=channel_middle, use_bias=use_bias, sn=False, scope='norm_2')
        x = lrelu(x, 0.2)
        x = conv(x, channels=channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_2')

        if channel_in != channels :
            x_init = spade(segmap, x_init, channels=channel_in, use_bias=use_bias, sn=False, scope='norm_shortcut')
            x_init = conv(x_init, channels=channels, kernel=1, stride=1, use_bias=False, sn=sn, scope='conv_shortcut')

        return x + x_init


def spade(segmap, x_init, channels, use_bias=True, sn=False, reuse=tf.compat.v1.AUTO_REUSE, scope=None) :
    with tf.compat.v1.variable_scope(scope, reuse=reuse) :
        #x = param_free_norm(x_init)
        x = batch_norm(x_init, scope="batch_norm")

        _, x_h, x_w, _ = x_init.get_shape().as_list()
        _, segmap_h, segmap_w, _ = segmap.get_shape().as_list()

        factor_h = segmap_h // x_h  # 256 // 4 = 64
        factor_w = segmap_w // x_w

        segmap_down = down_sample(segmap, factor_h, factor_w)

        segmap_down = conv(segmap_down, channels=128, kernel=5, stride=1, pad=2, use_bias=use_bias, sn=sn, scope='conv_128')
        segmap_down = relu(segmap_down)

        segmap_gamma = conv(segmap_down, channels=channels, kernel=5, stride=1, pad=2, use_bias=use_bias, sn=sn, scope='conv_gamma')
        segmap_beta = conv(segmap_down, channels=channels, kernel=5, stride=1, pad=2, use_bias=use_bias, sn=sn, scope='conv_beta')

        x = x * (1 + segmap_gamma) + segmap_beta

        return x

def cspade_resblock(context, segmap, x_init, channels, use_bias=True, sn=False, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    channel_in = x_init.get_shape().as_list()[-1]
    channel_middle = min(channel_in, channels)

    with tf.compat.v1.variable_scope(scope, reuse=reuse) :
        x = cspade(context, segmap, x_init, channel_in, use_bias=use_bias, sn=False, scope='norm_1')
        x = lrelu(x, 0.2)
        x = conv(x, channels=channel_middle, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_1')

        x = cspade(context, segmap, x, channels=channel_middle, use_bias=use_bias, sn=False, scope='norm_2')
        x = lrelu(x, 0.2)
        x = conv(x, channels=channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_2')

        if channel_in != channels :
            x_init = cspade(context, segmap, x_init, channels=channel_in, use_bias=use_bias, sn=False, scope='norm_shortcut')
            x_init = conv(x_init, channels=channels, kernel=1, stride=1, use_bias=False, sn=sn, scope='conv_shortcut')

        return x + x_init


def cspade(context, segmap, x_init, channels, use_bias=True, sn=False, reuse=tf.compat.v1.AUTO_REUSE, scope=None) :
    with tf.compat.v1.variable_scope(scope, reuse=reuse) :
        #x = param_free_norm(x_init)
        x = batch_norm(x_init, scope="batch_norm")

        _, x_h, x_w, _ = x_init.get_shape().as_list()
        _, segmap_h, segmap_w, _ = segmap.get_shape().as_list()

        factor_h = segmap_h // x_h  # 256 // 4 = 64
        factor_w = segmap_w // x_w

        context_gamma = fully_connected(context, units=channels, use_bias=use_bias, sn=sn, scope='linear_gamma')
        context_beta = fully_connected(context, units=channels, use_bias=use_bias, sn=sn, scope='linear_beta')

        context_shape = [-1, 1, 1, channels]
        context_gamma = tf.reshape(context_gamma, context_shape)
        context_beta = tf.reshape(context_beta, context_shape)

        segmap_down = down_sample(segmap, factor_h, factor_w)

        segmap_down = conv(segmap_down, channels=128, kernel=5, stride=1, pad=2, use_bias=use_bias, sn=sn, scope='conv_128')
        segmap_down = relu(segmap_down)

        segmap_gamma = conv(segmap_down, channels=channels, kernel=5, stride=1, pad=2, use_bias=use_bias, sn=sn, scope='conv_gamma')
        segmap_beta = conv(segmap_down, channels=channels, kernel=5, stride=1, pad=2, use_bias=use_bias, sn=sn, scope='conv_beta')

        x = x * (1 + (context_gamma + segmap_gamma)) + (context_beta + segmap_beta)

        return x

def cprogressive_resblock(context, segmap, x_init, channels, use_bias=True, sn=False, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    channel_in = x_init.get_shape().as_list()[-1]
    channel_middle = min(channel_in, channels)

    with tf.compat.v1.variable_scope(scope, reuse=reuse) :
        x = cprogressive(context, segmap, x_init, channel_in, use_bias=use_bias, sn=False, scope='norm_1')
        x = lrelu(x, 0.2)
        x = conv(x, channels=channel_middle, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_1')

        x = cprogressive(context, segmap, x, channels=channel_middle, use_bias=use_bias, sn=False, scope='norm_2')
        #x = adain(context, x, channels=channel_middle, use_bias=use_bias, sn=False, scope='norm_2')
        x = lrelu(x, 0.2)
        x = conv(x, channels=channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_2')

        if channel_in != channels :
            x_init = cprogressive(context, segmap, x_init, channels=channel_in, use_bias=use_bias, sn=False, scope='norm_shortcut')
            #x_init = adain(context, x_init, channels=channel_in, use_bias=use_bias, sn=False, scope='norm_shortcut')
            x_init = conv(x_init, channels=channels, kernel=1, stride=1, use_bias=False, sn=sn, scope='conv_shortcut')

        return x + x_init


def cprogressive(context, segmap, x_init, channels, use_bias=True, sn=False, reuse=tf.compat.v1.AUTO_REUSE, scope=None) :
    with tf.compat.v1.variable_scope(scope, reuse=reuse) :
        #x = param_free_norm(x_init)
        x = batch_norm(x_init, scope="batch_norm")

        _, x_h, x_w, _ = x_init.get_shape().as_list()
        _, segmap_h, segmap_w, _ = segmap.get_shape().as_list()

        factor_h = segmap_h // x_h  # 256 // 4 = 64
        factor_w = segmap_w // x_w

        context_gamma = fully_connected(context, units=channels, use_bias=use_bias, sn=sn, scope='linear_gamma')
        context_beta = fully_connected(context, units=channels, use_bias=use_bias, sn=sn, scope='linear_beta')

        context_shape = [-1, 1, 1, channels]
        context_gamma = tf.reshape(context_gamma, context_shape)
        context_beta = tf.reshape(context_beta, context_shape)

        if factor_h != 1 or factor_w != 1:
            segmap_down = down_sample(segmap, factor_h, factor_w)
        else:
            segmap_down = segmap

        #segmap_down = conv(segmap_down, channels=channels, kernel=5, stride=1, pad=2, use_bias=use_bias, sn=sn, scope='preconv')
        #segmap_down = relu(segmap_down)

        segmap_gamma = conv(segmap_down, channels=channels, kernel=5, stride=1, pad=2, use_bias=use_bias, sn=sn, scope='conv_gamma')
        segmap_beta = conv(segmap_down, channels=channels, kernel=5, stride=1, pad=2, use_bias=use_bias, sn=sn, scope='conv_beta')

        x = x * (1 + (context_gamma + segmap_gamma)) + (context_beta + segmap_beta)
        #x = (x * (1 + context_gamma) + context_beta) + segmap_beta

        return x

##################################################################################
# Sampling
##################################################################################

def resize(x,height=256, width=256) :
    return tf.image.resize(x, size=[height, width], method=tf.image.ResizeMethod.BILINEAR)

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.BILINEAR)

def down_sample(x, scale_factor_h, scale_factor_w) :
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h // scale_factor_h, w // scale_factor_w]

    return tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def down_sample_avg(x, scale_factor=2) :
    return tf.compat.v1.layers.average_pooling2d(x, pool_size=3, strides=scale_factor, padding='SAME')


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)

def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

def softmax(x):
    return tf.nn.softmax(x)

##################################################################################
# Normalization function
##################################################################################

def param_free_norm(x, epsilon=1e-5) :
    mean, var = tf.nn.moments(x=x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(var + epsilon)
    result = (x - mean) * inv

def param_free_batch_norm(x, epsilon=1e-5, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        mean, var = tf.nn.moments(x, range(len(x.get_shape())-1), keep_dims=True)
        #inv = tf.math.rsqrt(var + epsilon)
        #result = (x - mean) * inv
        result = tf.nn.batch_normalization(x, mean, var, tf.zeros_like(mean), tf.ones_like(var), epsilon)
    return result

def instance_norm(x, epsilon=1e-5, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        mean, var = tf.nn.moments(x=x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(var + epsilon)
        shape = mean.get_shape().as_list()
        offset = get_trainable_variable("offset", initializer=np.zeros(shape, dtype='float32'))
        scale = get_trainable_variable("scale", initializer=np.ones(shape, dtype='float32'))
        result = (x - mean) * inv * scale + offset
    return result

def batch_norm(x, epsilon=1e-5, reuse=tf.compat.v1.AUTO_REUSE, scope=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        mean, var = tf.nn.moments(x=x, axes=list(range(len(x.get_shape())-1)), keepdims=True)
        shape = mean.get_shape().as_list()
        offset = get_trainable_variable("offset", initializer=np.zeros(shape, dtype='float32'))
        scale = get_trainable_variable("scale", initializer=np.ones(shape, dtype='float32'))
        #inv = tf.math.rsqrt(var + epsilon)
        #result = (x - mean) * inv * scale + offset
        result = tf.nn.batch_normalization(x, mean, var, offset, scale, epsilon)
    return result

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.compat.v1.get_variable("u", [1, w_shape[-1]], initializer=tf.compat.v1.random_normal_initializer(), trainable=False, aggregation=tf.compat.v2.VariableAggregation.ONLY_FIRST_REPLICA)#, synchronization=tf.compat.v2.VariableSynchronization.ON_READ)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(a=w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(a=u_hat))

    #with tf.control_dependencies([u.assign(u_hat)]):
    #    w_norm = w / sigma
    #    w_norm = tf.reshape(w_norm, w_shape)
    u.assign(u_hat)
    w_norm = w / sigma
    w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


##################################################################################
# Loss function
##################################################################################

def L2_loss(x, y):
    loss = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.square(x - y), axis=-1))

    return loss

def L2_mean_loss(x, y):
    loss = tf.reduce_mean(input_tensor=tf.reduce_mean(input_tensor=tf.square(x - y), axis=-1))

    return loss

def L1_loss(x, y):
    loss = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.abs(x - y), axis=-1))

    return loss

def L1_mean_loss(x, y):
    loss = tf.reduce_mean(input_tensor=tf.reduce_mean(input_tensor=tf.abs(x - y), axis=-1))

    return loss

def discriminator_scores(real, fake):
    real_scores = []
    fake_scores = []

    for i in range(len(fake)):
        real_score = tf.reduce_mean(input_tensor=real[i][-1])
        fake_score = tf.reduce_mean(input_tensor=fake[i][-1])

        real_scores.append(real_score)
        fake_scores.append(fake_score)

    real_score = tf.reduce_mean(input_tensor=real_scores)
    fake_score = tf.reduce_mean(input_tensor=fake_scores)
    return real_score, fake_score

def discriminator_loss(loss_func, real, fake):
    loss = []
    real_loss = 0
    fake_loss = 0

    for i in range(len(fake)):
        if loss_func == 'lsgan':
            real_loss = tf.reduce_mean(input_tensor=tf.math.squared_difference(real[i][-1], 1.0))
            fake_loss = tf.reduce_mean(input_tensor=tf.square(fake[i][-1]))

        if loss_func == 'gan' or loss_func == 'dragan':
            real_loss = tf.reduce_mean(
                input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i][-1]), logits=real[i][-1]))
            fake_loss = tf.reduce_mean(
                input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i][-1]), logits=fake[i][-1]))

        if loss_func == 'hinge':
            # real_loss = tf.reduce_mean(relu(1.0 - real[i][-1]))
            # fake_loss = tf.reduce_mean(relu(1.0 + fake[i][-1]))
            real_loss = -tf.reduce_mean(input_tensor=tf.minimum(real[i][-1] - 1, 0.0))
            fake_loss = -tf.reduce_mean(input_tensor=tf.minimum(-fake[i][-1] - 1, 0.0))

        if loss_func.__contains__('wgan'):
            real_loss = -tf.reduce_mean(input_tensor=real[i][-1])
            fake_loss = tf.reduce_mean(input_tensor=fake[i][-1])

        loss.append(real_loss + (fake_loss))

    return tf.reduce_mean(input_tensor=loss)

def generator_loss(loss_func, fake):
    loss = []
    fake_loss = 0

    for i in range(len(fake)):
        if loss_func == 'lsgan':
            fake_loss = tf.reduce_mean(input_tensor=tf.math.squared_difference(fake[i][-1], 1.0))

        if loss_func == 'gan' or loss_func == 'dragan':
            fake_loss = tf.reduce_mean(
                input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake[i][-1]), logits=fake[i][-1]))

        if loss_func == 'hinge':
            # fake_loss = -tf.reduce_mean(relu(fake[i][-1]))
            fake_loss = -tf.reduce_mean(input_tensor=fake[i][-1])

        if loss_func.__contains__('wgan'):
            fake_loss = -tf.reduce_mean(input_tensor=fake[i][-1])

        loss.append(fake_loss)

    return tf.reduce_mean(input_tensor=loss)

def feature_stop_gradient(features):
    result = []
    for l in features:
        result.append(list([tf.stop_gradient(t) for t in l]))
    return result

def feature_loss(real, fake) :

    loss = []

    for i in range(len(fake)) :
        intermediate_loss = 0
        for j in range(len(fake[i]) - 1) :
            intermediate_loss += L1_mean_loss(real[i][j], fake[i][j])
        loss.append(intermediate_loss)

    return tf.reduce_mean(input_tensor=loss)

def z_sample(mean, logvar):
    eps = tf.random.normal(tf.shape(input=mean), mean=0.0, stddev=1.0, dtype=tf.float32)

    return mean + tf.exp(logvar * 0.5) * eps

def ce_loss(p,q_logits):
    return tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(p), logits=q_logits))

def kl_loss(mean, logvar):
    # shape : [batch_size, channel]
    k = mean.get_shape()[-1]
    loss = 0.5 * tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.square(mean) + tf.exp(logvar) - 1 - logvar, axis=-1)) / int(k)
    # loss = tf.reduce_mean(loss)

    return loss

def kl_loss2(mean1, logvar1, mean2, logvar2):
    # shape : [batch_size, channel]
    k = mean1.get_shape()[-1]
    loss = 0.5 * tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.square(mean2-mean1)/tf.exp(logvar2) + tf.exp(logvar1-logvar2)  - 1 + (logvar2 - logvar1), axis=-1)) / int(k)
    # loss = tf.reduce_mean(loss)

    return loss

def gaussian_loss(x, mean, logvar):
    k = mean.get_shape()[-1]
    pi = tf.constant(math.pi)
    loss = 0.5*tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.square(x - mean) / tf.exp(logvar) + logvar + tf.math.log(2*pi), axis=-1)) / int(k)
    # loss = tf.reduce_mean(loss)

    return loss

def negent_loss(mean, logvar):
    # shape : [batch_size, channel]
    k = mean.get_shape()[-1]
    pi = tf.constant(math.pi)
    loss = -0.5 * tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=logvar + tf.math.log(2*pi) + 1, axis=-1)) / int(k)
    # loss = tf.reduce_mean(loss)

    return loss

def gaussian_wasserstein2_loss(mean1, logvar1, mean2, logvar2):
    k = mean1.get_shape()[-1]
    loss = 0.5 * tf.reduce_mean(input_tensor=tf.reduce_sum((mean1-mean2)**2 + (tf.exp(logvar1*0.5) - tf.exp(logvar2*0.5))**2, axis=-1)) / int(k)  
    return loss

def regularization_loss(scope_name) :
    """
    If you want to use "Regularization"
    g_loss += regularization_loss('generator')
    d_loss += regularization_loss('discriminator')
    """
    collection_regularization = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)

    loss = []
    for item in collection_regularization :
        if scope_name in item.name :
            loss.append(item)

    return tf.reduce_sum(input_tensor=loss)
