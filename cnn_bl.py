#!/usr/bin/env python3
# coding=utf-8
__author__ = 'BAILEY'

import tensorflow as tf
import tensorrec
import numpy as np


class DocCNN(object):

    def __init__(self, document_num, document_length, embedding_size, num_classes, filter_sizes, num_filters,
                 full_layers, l2_reg_lambda=0.0):
        with tf.name_scope("inputs"):
            self.input_x = tf.placeholder(tf.float32, [None, document_num, document_length, embedding_size],
                                          name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        ##        l2_loss = tf.constant(0.0)

        self.input_x_trans = tf.transpose(self.input_x, perm=[1, 0, 2, 3])
        self.x_ins = tf.gather_nd(self.input_x_trans, [0])
        self.x_in0 = tf.expand_dims(self.x_ins, -1)
        # x_in0 shape=[batch,document_length, embedding_size]
        score = []
        with tf.variable_scope("final-score") as scope:
            for i in range(document_num - 1):
                x_in_tmp = tf.expand_dims(tf.gather_nd(self.input_x_trans, [i + 1]), -1)
                # shape=[batch,document_length, embedding_size]
                [score_single, l2_loss] = cnn_single_cal(self.x_in0, x_in_tmp, document_length, embedding_size,
                                                         num_classes, filter_sizes, num_filters,
                                                         full_layers, self.dropout_keep_prob, l2_reg_lambda=0.0)
                self.loss_tmp=l2_loss
                scope.reuse_variables()
                score.append(score_single)
            self.final_scores = tf.concat(score, 1, name="final_scores")
            # calculate the final one in the scores
        with tf.name_scope("final_predict"):
            self.final_predictions = tf.argmax(self.final_scores, axis=1, name="predictions")

            # calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.final_scores, labels=self.input_y)
            self.loss = tf.add(tf.reduce_mean(losses), self.loss_tmp * l2_reg_lambda, name="loss")
            # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.final_predictions, tf.argmax(self.input_y, 1))
            self.accuracy =tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            # argmax is to find the index!


##      Shape must be rank 4 but is rank 3 for 'final-score/conv-pool-filter/conv-maxpool-3/conv' (op: 'Conv2D') with input shapes: [?,3600,300], [3,300,1,128].  

def filters(document_length, filter_sizes, embedding_size, num_filters, x_in):
    num_filters_total = num_filters * len(filter_sizes)
    pool_Q = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.get_variable(name="W", shape=filter_shape, initializer=tf.truncated_normal_initializer)
            b = tf.get_variable(name="b", shape=[num_filters], initializer=tf.constant_initializer)
            conv = tf.nn.conv2d(
                x_in,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, document_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="pool")
            # pooled=[batch,1,1,num_filters]
            pool_Q.append(pooled)
            # pooled_outputs=[3,batch,1,1,num_filters]
        # tf.concat=[batch,1,1,num_filters*3][23,1,1,384]
    h_pool_Q = tf.concat(pool_Q, 3)
    # tf.reshape=[batch,number_filters_total][23,384]
    h_pool_flat_Q = tf.reshape(h_pool_Q, [-1, num_filters_total])
    return h_pool_flat_Q


##    input_x1 = tf.placeholder(tf.float32, [None, document_length, embedding_size], name="input_x1")
##    input_x2 = tf.placeholder(tf.float32, [None, document_length, embedding_size], name="input_x2")
##    input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
##    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")   

def cnn_single_cal(input_x1, input_x2, document_length, embedding_size, num_classes, filter_sizes, num_filters,
                   full_layers, dropout_keep_prob, l2_reg_lambda=0.0):
    num_filters_total = num_filters * len(filter_sizes)

    # keep track of L2 loss
    l2_loss = tf.constant(0.0)

    with tf.variable_scope("conv-pool-filter") as scope:
        x_convol1 = filters(document_length, filter_sizes, embedding_size, num_filters, input_x1)
        scope.reuse_variables()
        x_convol2 = filters(document_length, filter_sizes, embedding_size, num_filters, input_x2)
        x_convol = tf.concat([x_convol1, x_convol2], 1)

    # x_convol.shape[batch,num_filters_total+2]
    with tf.name_scope("dropout"):
        x_drop = tf.nn.dropout(x_convol, dropout_keep_prob)

    with tf.variable_scope("fully-connected-layer"):
        W = tf.get_variable(name="W", shape=[num_filters_total * 2, full_layers],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable(name="b", shape=[full_layers], initializer=tf.constant_initializer)
        middle = tf.nn.xw_plus_b(x_drop, W, b, name="middle")
        middle_out = tf.nn.tanh(middle, name="middle_out")
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)

    with tf.variable_scope("output"):
        W = tf.get_variable(name="W", shape=[full_layers, 1],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable(name="b", shape=[1], initializer=tf.constant_initializer(0.1))
        score_q = tf.nn.xw_plus_b(middle_out, W, b, name="score_q")
        l2_loss += tf.nn.l2_loss(W)

    return score_q, l2_loss