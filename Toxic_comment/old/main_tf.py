########################################################################
# Kaggle Mercari Price Suggestion Challenge (Tensorflow implementation)
# http://www.kaggle.com/c/mercari-price-suggestion-challenge
########################################################################

import time
# import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import re

from imblearn.over_sampling import RandomOverSampler
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer

# Parameters
EPOCHS = 1
BATCH_SIZE = 16


class Model:

    def __init__(self, input_dim, vocab_size, iterator, params, mode):
        self.params = params
        self.mode = mode

        # Define iterator
        self.next_x_text_ids, self.next_x_features, self.next_y = iterator.get_next()

        print(self.next_x_text_ids.shape)
        print(self.next_x_features.shape)
        print(self.next_y.shape)  # Expect one hot vector

        logits = self.model_fcn(input_dim, vocab_size, self.next_x_text_ids, self.next_x_features, params, mode=self.mode)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.next_y, logits=logits)

        self.predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if self.mode == "TRAIN":
            self.training_op = tf.train.AdagradOptimizer(learning_rate=params['learning_rate']).minimize(self.loss)

        elif self.mode == "EVAL":
            self.eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(self.next_y, predictions=self.predictions['classes'])
            }

        elif self.mode == "INFER":
            self.ids, self.next_x = iterator.get_next()
            self.y_pred = self.model_fcn(input_dim, self.next_x, params, mode=self.mode)
        else:
            raise ValueError("Mode should be TRAIN/EVAL/INFER")

        self.saver = tf.train.Saver()

    def model_fcn(self, input_dim, vocab_size, next_x_text_ids, next_x_features, params, mode):
        # TODO: Build the tensorflow model here, return prediction
        with tf.variable_scope("embedding"):
            # Embedding matrix
            embedding = tf.get_variable("embedding_weight", shape=[vocab_size, params['embedding_size']])

            # Embedding lookup
            embedded_input = tf.nn.embedding_lookup(embedding, next_x_text_ids)

        with tf.variable_scope("rnn"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(params['lstm_size'])
            outputs, states = tf.nn.dynamic_rnn(lstm_cell, embedded_input, dtype=tf.float32)

        with tf.variable_scope("combine"):
            output_t = tf.transpose(outputs, [1, 0, 2])
            last = tf.gather(output_t, int(output_t.get_shape()[0]) - 1)
            merged_inputs = tf.concat([last, next_x_features], axis=1)

        with tf.variable_scope("dense1"):
            dense1 = tf.layers.dense(merged_inputs, units=192, activation=tf.nn.relu, name="dense1")

        with tf.variable_scope("dropout1"):
            dropout1 = tf.layers.dropout(dense1, rate=params['dropout_rate'], name="dropout1")

        with tf.variable_scope("dense2"):
            dense2 = tf.layers.dense(dropout1, units=64, activation=tf.nn.relu)

        with tf.variable_scope("pred"):
            logits = tf.layers.dense(inputs=dense2, units=2)

        return logits

    def train(self, sess):
        _, loss = sess.run([self.training_op, self.loss])
        return loss

    def eval(self, sess):
        loss = sess.run(self.loss)
        eval_metrics = sess.run(self.eval_metrics_ops)
        accuracy = eval_metrics['accuracy']
        return loss, accuracy

    def predict(self, sess):
        y_pred = sess.run(self.y_pred)
        return self.ids, y_pred


class SelectColumnsTransfomer(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, **transform_params):
        return X[self.columns]

    def fit(self, X, y=None, **fit_params):
        return self


def to_record(df):
    return df.to_dict(orient='records')


def load_data():
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_test


def preprocess(df_train, df_test) -> tuple:
    # TODO: Preproceses pipelines and return cleaned dataframe
    df_train_cleaned = _preprocess(df_train)
    df_test_cleaned = _preprocess(df_test)

    # Preview dataset
    print("Preprocess done. Preview the dataset...")
    print("df_train_cleaned")
    print(df_train_cleaned.head(n=10))
    print("df_test_cleaned")
    print(df_test_cleaned.head(n=10))

    max_text_length = max(df_train_cleaned["text_length"].max(),
                          df_test_cleaned["text_length"].max())

    # input("Press Enter to continue...")

    # transformed_datasets, vocab_size = transform_dataset(df_train_cleaned, df_test_cleaned, max_text_length)

    return df_train_cleaned, df_test_cleaned, max_text_length


def _preprocess(df):
    # Feature engineering - Proportion of uppercase text
    df = df.assign(text_length=df['comment_text'].apply(lambda x: len(word_tokenize(x))))
    df = df.assign(char_length=df['comment_text'].apply(lambda x: len(x)))
    df = df.assign(num_cap_char=df['comment_text'].apply(lambda x: sum(map(str.isupper, x))))
    df = df.assign(upper_prop=df.num_cap_char / df.char_length)

    # Convert the comment text to lower case
    df.loc[:, 'comment_text'] = df['comment_text'].astype(str).str.lower()

    # Text normalisation
    df.loc[:, 'comment_text'] = df['comment_text'].astype(str).str.replace(
        re.compile(r"[^A-Za-z0-9!\s]+|[\n]+|[\s]{2,}"), "")

    return df


def resample(X, y):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_sample(X, y)
    return X_resampled, y_resampled


def transform_dataset(df, cls_type, max_text_length=1000, train=True, vocab_processor=None):

    print("Stage 2 - Transforming dataset...")
    features = ['text_length', 'upper_prop']
    raw_features_x = df.as_matrix(columns=features)

    # Vocab Processor
    print("Extracting vocabs...")
    if vocab_processor is None:
        print("No Vocab Processor is provided...Creating one")
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length=max_text_length,
                                                                             min_frequency=5)
    else:
        print("Using existing vocab processor")
        vocab_processor = vocab_processor

    vocab_processor.fit(df["comment_text"])
    ids_x = np.array(list(vocab_processor.transform(df["comment_text"])))

    if train:
        y = df.loc[:, cls_type]

        # Resample dataset
        print("Resampling dataset for %s" % cls_type)
        resampled_x_features, resampled_y = resample(raw_features_x, y)
        resampled_ids_x, _ = resample(ids_x, y)

        # Split dataset
        print("Spliting dataset for %s" % cls_type)
        train_X_features, val_X_features, train_y, val_y = \
            train_test_split(resampled_x_features, resampled_y)

        train_X_comment_text, val_X_comment_text, _, _ = \
            train_test_split(resampled_ids_x, resampled_y)

        del ids_x
        del resampled_ids_x
        del resampled_y

        gc.collect()

        return (train_X_features,
                train_X_comment_text,
                train_y,
                val_X_features,
                val_X_comment_text,
                val_y,
                vocab_processor)

    else:
        return ids_x


def train_iterator_generator(features, comment_text, labels, batch_size=2048):
    """An input function for training"""
    # Convert the inputs to a Dataset.

    # sparse_features = tf.sparse_reorder(convert_sparse_matrix_to_sparse_tensor(features))
    inputs = (tf.cast(comment_text, dtype=tf.int32),
              tf.cast(features, dtype=tf.float32),
              tf.one_hot(tf.convert_to_tensor(labels), depth=2, on_value=1., off_value=0., dtype=tf.float32))
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_initializable_iterator()


# def eval_iterator_generator(features, labels=None, test_ids=None, batch_size=2048):
#     """An input function for evaluation or prediction"""
#
#     if labels is not None:
#         # No labels, use only features.
#         inputs = (tf.cast(tf.sparse_reorder(convert_sparse_matrix_to_sparse_tensor(features)), dtype=tf.float32),
#                   tf.cast(tf.reshape(tf.convert_to_tensor(labels), shape=[-1, 1]), dtype=tf.float32))
#     else:
#         inputs = (
#             tf.reshape(tf.convert_to_tensor(test_ids), shape=[-1, 1]),
#             tf.cast(tf.sparse_reorder(convert_sparse_matrix_to_sparse_tensor(features)), dtype=tf.float32))
#
#     # Convert the inputs to a Dataset.
#     dataset = tf.data.Dataset.from_tensor_slices(inputs)
#
#     # Batch the examples
#     assert batch_size is not None, "batch_size must not be None"
#     dataset = dataset.batch(batch_size)
#
#     # Return the read end of the pipeline.
#     return dataset.make_initializable_iterator()


def main(argy):
    gc.collect()

    # TODO: 1. Load dataset
    # TODO: 2. Pre-process dataset

    # Fetch the data
    df_train, df_test = load_data()
    df_train_cleaned, df_test_cleaned, max_text_length = preprocess(df_train, df_test)

    # Transform the dataset
    classifier_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Transform the dataset and run the model one by one

    train_X_features, \
    train_X_comment_text, \
    train_y, \
    val_X_features, \
    val_X_comment_text, \
    val_y, \
    vocab_processor = transform_dataset(df_train_cleaned, cls_type='toxic')

    vocab_dict = vocab_processor.vocabulary_._mapping
    vocab_size = len(vocab_dict)

    print(train_X_features.shape)
    print(train_X_comment_text.shape)
    print(train_y.shape)

    print(np.sum(np.where(train_y == 0)))
    print(np.sum(np.where(train_y == 1)))

    # for key, classifier in datasets.items():
    #
    #     print(key)
    #     print(type(classifier))
    #
    #     for ds_key, dataset in classifier[key]:
    #         print(ds_key)
    #         print(dataset[ds_key].shape)
    #
    # print(datasets)
    # print(max_text_length)
    # print(n_words)

    # TODO: 3. Build train/eval/infer Graph

    train_graph = tf.Graph()
    eval_graph = tf.Graph()
    infer_graph = tf.Graph()

    hparams = {'learning_rate': 0.003,
               'dropout_rate': 0.0,
               'lstm_size': 100,
               'embedding_size': 300}


    INPUT_SIZE = train_X_features.shape[0]
    INPUT_features_DIM = train_X_features.shape[1]
    INPUT_text_DIM = train_X_comment_text.shape[1]

    # print("BATCH SIZE: %s, INPUT SIZE: %s, INPUT DIM: %s" % (BATCH_SIZE, INPUT_SIZE, INPUT_DIM))
    #
    steps = (INPUT_SIZE // BATCH_SIZE)
    #
    # print("Total steps of training: %s" % (steps * EPOCHS))
    #
    # print("Parameters: %s" % params)
    #
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 44
    config.inter_op_parallelism_threads = 44

    #
    # # TODO: 4. Execute the graph and train the model
    #
    with train_graph.as_default():
        print("Building trianing graph")
        train_iterator = train_iterator_generator(train_X_features,
                                                  train_X_comment_text,
                                                  train_y,
                                                  batch_size=BATCH_SIZE)
        train_model = Model(input_dim=INPUT_features_DIM,
                            vocab_size=vocab_size,
                            iterator=train_iterator,
                            params=hparams,
                            mode="TRAIN")
        initializer = tf.global_variables_initializer()
    #
    # with eval_graph.as_default():
    #     print("Building evaluation graph")
    #     eval_iterator = eval_iterator_generator(val_X, val_y, batch_size=BATCH_SIZE)
    #     eval_model = Model(input_dim=INPUT_DIM, iterator=eval_iterator, params=params, mode="EVAL")
    #
    # with infer_graph.as_default():
    #     print("Building inference graph")
    #     test_ids = df_test['test_id']
    #     infer_iterator = eval_iterator_generator(test_X, test_ids=test_ids, batch_size=BATCH_SIZE)
    #     infer_model = Model(input_dim=INPUT_DIM, iterator=infer_iterator, params=params, mode="INFER")
    #
    train_sess = tf.Session(graph=train_graph, config=config)
    # eval_sess = tf.Session(graph=eval_graph, config=config)
    # infer_sess = tf.Session(graph=infer_graph, config=config)
    train_sess.run(initializer)
    train_sess.run(train_iterator.initializer)

    #
    # checkpoints_path = "./checkpoints/"
    #
    t0 = time.time()
    #
    for epoch in range(1, EPOCHS + 1):
        for step in range(steps):
            try:
                train_loss = train_model.train(train_sess)

                if step % 100 == 0:
                    t1 = time.time()
                    print("Epoch %s - Step: %s/%s train loss: {%s} - Time used: %s" % (epoch,
                                                                                       step,
                                                                                       steps,
                                                                                       train_loss,
                                                                                       (t1 - t0)))

                #     checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=i)
                #     infer_model.saver.restore(infer_sess, checkpoint_path)
                #     infer_sess.run(infer_iterator.initializer, feed_dict={infer_inputs: infer_input_data})
                #     while data_to_infer:
                #         infer_model.infer(infer_sess)
            except tf.errors.OutOfRangeError:
                break
    #
    #     if epoch > 1:
    #         # print("Steps:%s loss: {%s}" % (steps, train_model.loss))
    #         checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=step * epoch,
    #                                                  write_meta_graph=False)
    #         eval_model.saver.restore(eval_sess, checkpoint_path)
    #         eval_sess.run(eval_iterator.initializer)
    #
    #         eval_loss = eval_model.eval(eval_sess)
    #
    #         t1 = time.time()
    #         print("Epoch %s - Step: %s/%s eval loss: {%s} - Time used: %s" % (epoch,
    #                                                                           step, steps, eval_loss, (t1 - t0)))
    #
    save_path = train_model.saver.save(train_sess, "./model.ckpt", write_meta_graph=False)
    #
    # train_sess.close()
    # eval_sess.close()
    #
    # infer_sess.run(infer_iterator.initializer)
    # infer_model.saver.restore(infer_sess, save_path)
    # result = []
    # while True:
    #     try:
    #         id, pred = infer_model.predict(infer_sess)
    #         print(zip(id, pred))
    #     except tf.errors.OutOfRangeError:
    #         break


if __name__ == "__main__":
    tf.app.run(main)