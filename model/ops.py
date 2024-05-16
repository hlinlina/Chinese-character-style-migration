# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

weight_init = tf.compat.v1.random_normal_initializer(mean= 0.0,stddev =0.02 )
weight_regularizer = tf.keras.regularizers.l2(l=0.5 * (0.0001))




def batch_norm(x, is_training, epsilon=1e-5, momentum=0.9, name="batch_norm"):
    return BatchNormalization(momentum=momentum, epsilon=epsilon, name=name)(x, training=is_training)


def conv2d(x, output_filters, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="conv2d"):
    with tf.compat.v1.variable_scope(scope):
        shape = x.get_shape().as_list()
        W = tf.compat.v1.get_variable('W', [kh, kw, shape[-1], output_filters],
                            initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
        Wconv = tf.nn.conv2d(x, filters=W, strides=[1, sh, sw, 1], padding='SAME')

        biases = tf.compat.v1.get_variable('b', [output_filters], initializer=tf.compat.v1.constant_initializer(0.0))
        Wconv_plus_b = tf.reshape(tf.nn.bias_add(Wconv, biases), Wconv.get_shape())

        return Wconv_plus_b

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv'):
    with tf.compat.v1.variable_scope(scope):
        if pad > 0 :
            if (kernel - stride) % 2 == 0:
                pad_top = pad
                pad_bottom = pad
                pad_left = pad
                pad_right = pad

            else:
                pad_top = pad
                pad_bottom = kernel - stride - pad_top
                pad_left = pad
                pad_right = kernel - stride - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn :
            shape = x.get_shape().as_list()
            w = tf.compat.v1.get_variable("kernel", [kernel, kernel, shape[-1], channels], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02), regularizer=tf.keras.regularizers.l2(l=0.5 * (0.0001)))
            x = tf.nn.conv2d(x, filters=w, strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                biases = tf.compat.v1.get_variable("bias", [channels], initializer=tf.compat.v1.constant_initializer(0.0))
                wconv_plus_b = tf.reshape(tf.nn.bias_add(x, biases), x.get_shape())

        else :
            wconv_plus_b = tf.compat.v1.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * (0.0001)),
                                 strides=stride, use_bias=use_bias)
        return wconv_plus_b


def deconv2d(x, output_shape, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="deconv2d"):
    with tf.compat.v1.variable_scope(scope):
        # filter : [height, width, output_channels, in_channels]
        input_shape = x.get_shape().as_list()
        W = tf.compat.v1.get_variable('W', [kh, kw, output_shape[-1], input_shape[-1]],
                            initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(x, W, output_shape=output_shape,
                                        strides=[1, sh, sw, 1], padding='SAME')

        biases = tf.compat.v1.get_variable('b', [output_shape[-1]], initializer=tf.compat.v1.constant_initializer(0.0))
        deconv_plus_b = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv_plus_b


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def relu(x):
    return tf.nn.relu(x)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

    return gap


def fc(x, output_size, stddev=0.02, scope="fc"):
    with tf.compat.v1.variable_scope(scope):
        shape = x.get_shape().as_list()
        W = tf.compat.v1.get_variable("W", [shape[1], output_size], tf.float32,
                            initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
        b = tf.compat.v1.get_variable("b", [output_size],
                            initializer=tf.compat.v1.constant_initializer(0.0))
        return tf.matmul(x, W) + b


def init_embedding(size, dimension, stddev=0.01, scope="embedding"):
    with tf.compat.v1.variable_scope(scope):
        return tf.compat.v1.get_variable("E", [size, 1, 1, dimension], tf.float32,
                               initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))


def conditional_instance_norm(x, ids, labels_num, mixed=False, scope="conditional_instance_norm"):
    with tf.compat.v1.variable_scope(scope):
        shape = x.get_shape().as_list()
        batch_size, output_filters = shape[0], shape[-1]
        scale = tf.compat.v1.get_variable("scale", [labels_num, output_filters], tf.float32, initializer=tf.compat.v1.constant_initializer(1.0))
        shift = tf.compat.v1.get_variable("shift", [labels_num, output_filters], tf.float32, initializer=tf.compat.v1.constant_initializer(0.0))

        mu, sigma = tf.nn.moments(x, [1, 2], keepdims=True)
        norm = (x - mu) / tf.sqrt(sigma + 1e-5)

        batch_scale = tf.reshape(tf.nn.embedding_lookup(params=scale, ids=ids), [batch_size, 1, 1, output_filters])
        batch_shift = tf.reshape(tf.nn.embedding_lookup(params=shift, ids=ids), [batch_size, 1, 1, output_filters])

        z = norm * batch_scale + batch_shift
        return z