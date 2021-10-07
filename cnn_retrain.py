#!/usr/bin/env python3
# coding=utf-8
__author__ = 'BAILEY'

import tensorflow as tf
import numpy as np
import os
from cnn_bl import DocCNN
from tensorflow.contrib import learn
import data_helpers
import time
import datetime

# parameters
# tf.flags.define global variables
# load para
# C:\Users\baili\PycharmProjects\CNN212 the orignal one is trained by sets_eval
# now I retrain it based on the sets_eval2

tf.flags.DEFINE_float("dev_sample_percentage", .1, "percentage of the training data to use for validation")
tf.flags.DEFINE_string("query_path","E:\\CNN212\\sets_eval2\\queries","query cases path")
tf.flags.DEFINE_string("docu_path","E:\\CNN212\\sets_eval2\\docus","documents cases path")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_size", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_integer("document_num", 12, "Number of documents for query (default: 11)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("full_layers", 60, "number of neurons of first fully connected layers")
tf.flags.DEFINE_integer("document_length", 3600, "maximum number of document length")
tf.flags.DEFINE_integer("num_classes", 11, "maximum number of document length")
# Training parameters


# Training parameters
tf.flags.DEFINE_integer("training_samples",100,"training set size")
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("checkpoint_dir", "E:\\CNN212\\runs\\1518437210\\checkpoints", "Checkpoint directory from training run")


FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


## Data Preparation
# ==================================================

# Load data
print("Loading data...")
# inputs are path of files
# x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
x, y=data_helpers.load_data_and_labels(FLAGS.query_path, FLAGS.docu_path, FLAGS.training_samples, FLAGS.document_num, FLAGS.document_length, FLAGS.embedding_size)
# ([training_samples,document_num, document_length, embedding_size])

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(x.shape[0]))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(x.shape[0]))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

del x, x_shuffled
#
# print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(x_train.shape[0], x_dev.shape[0]))


# Training
# ==================================================
# checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
checkpoint_file="E:\\CNN212\\runs\\1518437210\\checkpoints\\model-600"
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        # find the trained grpah is used to restart training or run inference from a saved graph
        saver.restore(sess, checkpoint_file)
        input_x = graph.get_operation_by_name("inputs/input_x").outputs[0]
        input_y = graph.get_operation_by_name("inputs/input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("inputs/dropout_keep_prob").outputs[0]
        # loss_nn=graph.get_operation_by_name("loss")
        accuracy_nn=graph.get_operation_by_name("accuracy/accuracy").outputs[0]
        train_op=graph.get_collection('train_op')[0]
        global_step = graph.get_operation_by_name("global_step").outputs[0]



        # this is the graph.collections=[ 'summaries', 'train_op', 'trainable_variables', 'variables']
        # Define Training procedure
        # global_step = tf.Variable(0, name="global_step", trainable=False)
        # optimizer = tf.train.AdamOptimizer(1e-3)
        # grads_and_vars = optimizer.compute_gradients(loss)
        # gradients of cnn.loss tf.Variable with tuples of (grad,vars)
        # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        # grad_summaries = []
        # # read all the tuples
        # for g, v in grads_and_vars:
        #     if g is not None:
        #         print(g)
        #         print(v.name)
        #         grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        #         # generate values with histograms
        #         sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        #         grad_summaries.append(grad_hist_summary)
        #         grad_summaries.append(sparsity_summary)
        # grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        # loss_summary = tf.summary.scalar("loss", loss)
        acc_summary = tf.summary.scalar("accuracy", accuracy_nn)

        # Train Summaries
        train_summary_op = tf.summary.merge([ acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([ acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        # vocab_processor.save(os.path.join(out_dir, "vocab"))
##
        # Initialize all variables
        # if we retrain the model, we don't need to initializer it.

        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                input_x: np.array(x_batch),
                input_y: np.array(y_batch),
                dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, accuracy = sess.run(
                [train_op, global_step, train_summary_op, accuracy_nn],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, acc {:g}".format(time_str, step, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                input_x: np.array(x_batch),
                input_y: np.array(y_batch),
                dropout_keep_prob: 1.0
            }
            step, summaries,  accuracy = sess.run(
                [global_step, dev_summary_op,  accuracy_nn],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, acc {:g}".format(time_str, step, accuracy))
            if writer:
                writer.add_summary(summaries, step)

##
        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            print(type(x_batch))
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
