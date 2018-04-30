"""Routine for reading data"""

import pandas as pd
import numpy as np
import tensorflow as tf
import csv

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
            'label': tf.FixedLenFeature((), dtype=tf.int64, default_value=0)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        labels_return = parsed_features.pop('label')

        return parsed_features, labels_return

    train_dataset = dataset.map(lambda x: _parse_function(x))

    return train_dataset


def build_vocab(export=False):
    # Create a dictionary and create a tensorflow table
    df = pd.read_csv('data/train.csv')
    all_symbols, _, _ = word_processing.extract_word_vocab(df['comment_text'].values)
    all_symbols = [x.encode('utf-8') for x in all_symbols]

    if export:
        with open('data/vocab.csv', 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(all_symbols)

    return all_symbols


def train_eval_input_fn(dataset, table, batch_size):
    """An input function for training"""

    # Map the comment_text into ids
    def split_string(feature, label):
        comment_text = tf.cast(feature['comment_text'], tf.string)
        sentence = tf.string_split([comment_text]).values
        ids = table.lookup(sentence)
        return ids, label

    dataset_map = dataset.map(split_string)

    # Shuffle, repeat, and batch the examples.
    dataset_map = dataset_map.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])))

    dataset_map = dataset_map.shuffle(1000)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset_map

