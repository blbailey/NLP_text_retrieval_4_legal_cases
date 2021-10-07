#!/usr/bin/env python3
# coding=utf-8
__author__ = 'BAILEY'

import numpy as np
import re
import itertools
from collections import Counter
import os
import jieba
import pickle
import codecs
from gensim.models import Word2Vec

model_w2v = Word2Vec.load("E:\\CNN212\\zh\\zh.bin")
model_w2v.init_sims(replace=False)


# query_path="C:\\Users\\baili\\PycharmProjects\\bailey\\sets1\\queries"
# docu_path = "C:\\Users\\baili\\PycharmProjects\\bailey\\sets1\\docus"
def read_path(query_path, docu_path):
    """we put all the query documents paths into a list[list] 2-dim list"""
    query_file_names = [files for (root, dirs, files) in os.walk(query_path)][0]
    query_file_prefix = [os.path.splitext(file)[0] for file in query_file_names]
    docu_file_paths = [os.path.join(docu_path, file) for file in query_file_prefix]
    all_paths = []
    for i, file in enumerate(query_file_names):
        single_path = []
        single_path.append(os.path.join(query_path, query_file_names[i]))
        for (root, dirs, files) in os.walk(docu_file_paths[i] + "\\pos"):
            for file in files:
                single_path.append(os.path.join(root, file))
        for (root, dirs, files) in os.walk(docu_file_paths[i] + "\\neg1"):
            for file in files:
                single_path.append(os.path.join(root, file))
        all_paths.append(single_path)
    return all_paths


# query_path="C:\\Users\\baili\\PycharmProjects\\bailey\\sets1\\queries"
# docu_path = "C:\\Users\\baili\\PycharmProjects\\bailey\\sets1\\docus"
# training_samples=25
# document_num=5
# document_length=3600
# embedding_size=300
def load_data_and_labels(query_path, docu_path, training_samples, document_num, document_length, embedding_size):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    it is required
    all the files are tokenized by jieba and separated by space.
    all the words in each documents are in model_w2v.wv.
    """
    # Load data from files
    all_paths = read_path(query_path, docu_path)
    # print(all_paths)
    # all the files in the documents are pre-processed by tokenized and save with space. we only read each word and all the words are in the model.wv.
    # read a file and read
    # document_length=[codecs.open("C:\\Users\\bailey\\PycharmProjects\\law\\cpws_tokenized\\117440517.txt", "rb","utf-8").read().split(" ")]

    # document_num, document_length, embedding_size
    all_mtx = np.zeros([training_samples, document_num, document_length, embedding_size])
    # all_labels = np.zeros([training_samples, document_num - 1])
    for k, path_single in enumerate(all_paths):
        # print(path_single)
        q_mtx = np.zeros([document_num, document_length, embedding_size])
        # print(path_single)
        for j, file in enumerate(path_single):
            print(file)
            d_mtx = np.zeros([document_length, embedding_size])
            d_s = codecs.open(file, "r", "utf-8").read().split(" ")
            if len(d_s)>document_length:
                d_s=d_s[0:document_length]
            for i, w in enumerate(d_s):
                # print(w)
                d_mtx[i] = model_w2v[w]
            q_mtx[j] = d_mtx
        all_mtx[k] = q_mtx

    all_y = np.zeros([training_samples, document_num - 1])
    all_y[:, 0] = 1.

    return all_mtx, all_y


#
# batch_size=25
# num_epoches=1

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset. data.shape=[[training_samples,document_num, document_length, embedding_size]]
    """
    data = np.array(data)
    data_size = len(data)

    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
