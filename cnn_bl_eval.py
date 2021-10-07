#!/usr/bin/env python3
# coding=utf-8
__author__ = 'BAILEY'


import tensorflow as tf
import numpy as np
import os
import data_helpers
from cnn_bl import DocCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("query_eval_path","C:\\Users\\baili\\PycharmProjects\\CNN212\\sets_copy_eval2\\queries","query cases path")
tf.flags.DEFINE_string("docu_eval_path","C:\\Users\\baili\\PycharmProjects\\CNN212\\sets_copy_eval2\\docus","documents cases path")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "C:\\Users\\baili\\PycharmProjects\\CNN212\\runs\\1518437210\\checkpoints", "Checkpoint directory from training run")
# tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


# Model Hyperparameters
tf.flags.DEFINE_integer("training_samples",100,"training set size")
tf.flags.DEFINE_integer("embedding_size", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_integer("document_num", 12, "Number of documents for query (default: 11)")
tf.flags.DEFINE_integer("document_length", 3600, "maximum number of document length")
tf.flags.DEFINE_integer("num_classes", 11, "maximum number of document length")
FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
x_test, y_test=data_helpers.load_data_and_labels(FLAGS.query_eval_path, FLAGS.docu_eval_path, FLAGS.training_samples, FLAGS.document_num, FLAGS.document_length, FLAGS.embedding_size)
# if FLAGS.eval_train:
#     x, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
#     y_test = np.argmax(y_test, axis=1)
# else:
#     x_raw = ["a masterpiece four years in the making", "everything is off."]
#     y_test = [1, 0]

# Map data into vocabulary
# vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
# vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
# x_test = np.array(list(vocab_processor.transform(x_raw)))
#
# print("\nEvaluating...\n")

# Evaluation
# ==================================================
# find checkpoint_dir is the directary where the variables were saved
# return the full path to the latest checkpoint or none if it is not found
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print(checkpoint_file)
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

        # Get the placeholders from the graph by name
        # tf.Operation.outputs: the list of tensor objects representing the outputs of this op
        # and take the first one. namely, take all the inputs!
        input_x = graph.get_operation_by_name("inputs/input_x").outputs[0]
        input_y = graph.get_operation_by_name("inputs/input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("inputs/dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("final_predict/predictions").outputs[0]
        scores = graph.get_operation_by_name("final-score/final_scores").outputs[0]
        # scores is a vector of [batch,num_classes]
        # predictions is the max of scores

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_scores = np.empty((1,50,11))

        for x_test_batch in batches:
            batch_predictions, batch_scores = sess.run([predictions,scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            all_scores = np.append(all_scores, [batch_scores],axis=0)
        all_scores=np.delete(all_scores, [0], axis=0)
        print(all_scores.shape)

# predictions return the location of the maximum value
# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == np.argmax(y_test,1)))
    print(all_predictions)
    print(y_test)
    print("Total number of test examples: {}".format(FLAGS.training_samples))
    print("Accuracy: {:g}".format(correct_predictions/float(FLAGS.training_samples)))
    # print("Score: {:g}".format(all_scores))

# Save the evaluation to a csv
scores_human_readable = np.column_stack(all_scores)
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "score_copy_eval.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(scores_human_readable)
