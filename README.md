# NLP_text_retrieval
A text retrieval algorithm is developed based on CNN structure for legal case query (in Chinese).
# CNN (DL for NLP)
A CNN based on algorithm is designed to retrieve the most semantically similar case for each query legal case.
! The attached dataset are tokenized! Unsupervised learning algorithms are used for semantic analysis to create a labelled dataset for CNN based algorithm.

1. cnn_bl.py builds the structure for the CNN based algorithm;
2. cnn_bl_eval.py evaluates the algorithm;
3. cnn_train.py presents the training process;

4. cnn_retrain.py presents how to retrain the already trained model;
5. data_helpers.py prepares the data in the required format, such as converting Chinese text into vectors for modeling;
6. tokenizer.py is to use word2vec, jieba to tokenize the Chinese text;


