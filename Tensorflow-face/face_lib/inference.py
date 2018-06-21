#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-6-14 下午11:19
@email: lph0729@163.com
"""
import tensorflow as tf


class Siamese(object):
    def __init__(self, size):
        self.x_1 = tf.placeholder(tf.float32, [None, size, size, 3])
        self.x_2 = tf.placeholder(tf.float32, [None, size, size, 3])
        self.x_3 = tf.placeholder(tf.float32, [None, size, size, 3])
        self.d_1 = tf.placeholder(tf.float32, [None, 128])
        self.d_2 = tf.placeholder(tf.float32, [None, 128])
        self.d_3 = tf.placeholder(tf.float32, [None, 128])
        self.keep_f = tf.placeholder(tf.float32)

        with tf.variable_scope("siamese") as scope:
            self.o_1 = self.network(self.x_1, self.keep_f)
            scope.reuse_variables()
            self.o_2 = self.network(self.x_2, self.keep_f)
            scope.reuse_variables()
            self.o_3 = self.network(self.x_3, self.keep_f)

        self.loss = self.loss_with_spring()

    def network(self, x, keep_f):
        with tf.variable_scope("conv_1"):
            # 输出图片大小：96*96
            conv_1 = self.cnn_layer(x, [3, 3, 3, 64], [64])
        with tf.variable_scope("conv_2"):
            # 输出图片大小：48*48
            conv_2 = self.cnn_layer(conv_1, [3, 3, 64, 64], [64])
            pool_1 = self.pool_layer(conv_2, 1.0)
        with tf.variable_scope("conv_3"):
            # 输出图片大小：48*48
            conv_3 = self.cnn_layer(pool_1, [3, 3, 64, 128], [128])
        with tf.variable_scope("conv_4"):
            # 输出图片大小：24*24
            conv_4 = self.cnn_layer(conv_3, [3, 3, 128, 128], [128])
            pool_2 = self.pool_layer(conv_4, 1.0)
        with tf.variable_scope("conv_5"):
            # 输出图片大小：24*24
            conv_5 = self.cnn_layer(pool_2, [3, 3, 128, 256], [256])
        with tf.variable_scope("conv_6"):
            # 输出图片大小：24*24
            conv_6 = self.cnn_layer(conv_5, [3, 3, 256, 256], [256])
        with tf.variable_scope("conv_7"):
            # 输出图片大小：12 * 12
            conv_7 = self.cnn_layer(conv_6, [3, 3, 256, 256], [256])
            pool_3 = self.pool_layer(conv_7, 1.0)
        with tf.variable_scope("conv_8"):
            # 输出的图片大小：12*12
            conv_8 = self.cnn_layer(pool_3, [3, 3, 256, 512], [512])
        with tf.variable_scope("conv_9"):
            # 输出的图片大小：12*12
            conv_9 = self.cnn_layer(conv_8, [3, 3, 512, 512], [512])
        with tf.variable_scope("conv_10"):
            # 输出的图片大小：6*6
            conv_10 = self.cnn_layer(conv_9, [3, 3, 512, 512], [512])
            pool_4 = self.pool_layer(conv_10, 1.0)
        with tf.variable_scope("conv_11"):
            # 输出的图片大小：6*6
            conv_11 = self.cnn_layer(pool_4, [3, 3, 512, 512], [512])
        with tf.variable_scope("conv_12"):
            # 输出的图片大小：6*6
            conv_12 = self.cnn_layer(conv_11, [3, 3, 512, 512], [512])
        with tf.variable_scope("conv_13"):
            # 输出的图片大小：3*3
            conv_13 = self.cnn_layer(conv_12, [3, 3, 512, 512], [512])
            pool_5 = self.pool_layer(conv_13, 1.0)  # shape=(?, 3, 3, 512)

        with tf.variable_scope("full_layer_1"):
            # 此时输入的图片大小为：3*3
            fl_1 = self.full_layer(pool_5, [3 * 3 * 512, 1024], [1024], keep_f, True)  # (?, 1024)
        with tf.variable_scope("full_layer_2"):
            # 全连接层2，此时输入为：512
            fl_2 = self.full_layer(fl_1, [1024, 512], [512], keep_f)
        with tf.variable_scope("full_layer_3"):
            # 全连接层3,此时输入为：128
            fl_3 = self.full_layer(fl_2, [512, 128], [128], 1.0)
            print("--------------fl_3,shape---------------------------------", fl_3, fl_3.shape)
        return fl_3

    @staticmethod
    def cnn_layer(input_image, filter, bias_shape):
        init = tf.truncated_normal_initializer(stddev=0.04)
        weights = tf.get_variable("cnn_weights", dtype=tf.float32, shape=filter, initializer=init)
        biases = tf.get_variable("cnn_bias", dtype=tf.float32,
                                 initializer=tf.constant(0.01, shape=bias_shape, dtype=tf.float32))
        conv = tf.nn.conv2d(input_image, weights, strides=[1, 1, 1, 1], padding="SAME")
        relu_out = tf.nn.relu(conv + biases)
        return relu_out

    @staticmethod
    def pool_layer(input_image, keep):
        pool_out = tf.nn.max_pool(input_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        drop = tf.nn.dropout(pool_out, keep)
        return drop

    @staticmethod
    def full_layer(input_image, kernel_shape, bias_shape, keep, reshape=False):
        init = tf.truncated_normal_initializer(stddev=0.04)
        weights = tf.get_variable("cnn_weights", dtype=tf.float32, shape=kernel_shape, initializer=init)
        biases = tf.get_variable("cnn_biases", dtype=tf.float32,
                                 initializer=tf.constant(0.01, shape=bias_shape, dtype=tf.float32))

        # 第一次input_image.shape: [?, 3, 3, 512], 第一次执行此条件后nput_image.shape: [?, 4608]
        if reshape:
            input_image = tf.reshape(input_image, [-1, 3 * 3 * 512])  # shape=(?, 4608)
        # print("--------------input_image-----------------------", input_image)
        relu_out = tf.nn.relu(tf.matmul(input_image, weights) + biases)
        drop = tf.nn.dropout(relu_out, keep)
        return drop

    def loss_with_spring(self):
        margin = 5.0
        anchor_output = self.o_1
        # print("----------------------", anchor_output.shape)
        positive_output = self.o_2
        negative_output = self.o_3

        d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1, name="d_pos")
        d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1, name="d_neg")

        losses = tf.maximum(0., margin + d_pos - d_neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        # print("-----------------losses---------------", losses.shape)
        # print("----------------loss------------------", loss.shape)
        return loss
