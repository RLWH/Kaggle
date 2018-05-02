"""Routine for reading data"""

import pandas as pd
import numpy as np
import tensorflow as tf

from utils import word_processing


def read_data(file_path):
    """
    A function that support reading data into pandas df format or TF Dataset format.
    Support csv, tsv as import
    :param file_path: A dictionary of file paths.
                      file_paths['train'] = train dataset, file_paths['test'] = Test dataset
    :param feature_cols: List
    :param label_cols: List
    :param sep: Separator
    :param mode: Using pandas df or using tf API
    :param save_as_tfrecord: Boolean
    :return:
    """

    dataset = tf.data.TFRecordDataset(file_path)

    def _parse_function(example_proto):
        features = {
            'comment_text': tf.FixedLenFeature((), dtype=tf.string, default_value=""),
            'toxic': tf.FixedLenFeature((), dtype=tf.int64, default_value=0),
            'severe_toxic': tf.FixedLenFeature((), dtype=tf.int64, default_value=0),
            'obscene': tf.FixedLenFeature((), dtype=tf.int64, default_value=0),
            'threat': tf.FixedLenFeature((), dtype=tf.int64, default_value=0),
            'insult': tf.FixedLenFeature((), dtype=tf.int64, default_value=0),
            'identity_hate': tf.FixedLenFeature((), dtype=tf.int64, default_value=0)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        comment_text = tf.cast(parsed_features['comment_text'], tf.string)
        label = tf.stack([tf.cast(parsed_features['toxic'], tf.float32),
                          tf.cast(parsed_features['severe_toxic'], tf.float32),
                          tf.cast(parsed_features['obscene'], tf.float32),
                          tf.cast(parsed_features['threat'], tf.float32),
                          tf.cast(parsed_features['insult'], tf.float32),
                          tf.cast(parsed_features['identity_hate'], tf.float32),
                          ])

        return comment_text, label

    train_dataset = dataset.map(lambda x: _parse_function(x))

    return train_dataset


def train_eval_input_fn(dataset, table, batch_size):
    """An input function for training"""

    # Map the comment_text into ids
    def split_string(comment_text, label):
        comment_text = tf.cast(comment_text, tf.string)
        sentence = tf.string_split([comment_text]).values
        ids = table.lookup(sentence)
        return ids, label

    dataset_map = dataset.map(split_string)

    # Shuffle, repeat, and batch the examples.
    dataset_map = dataset_map.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None]),
                                                                      tf.TensorShape([6])))

    dataset_map = dataset_map.shuffle(1000)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset_map

