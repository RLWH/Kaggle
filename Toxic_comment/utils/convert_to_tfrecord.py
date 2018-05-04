"""
An utility file that convert dataset to TFRecord
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split

save_dir = "../data"

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _dtype_feature(ndarray):
    """match appropriate tf.train.Feature class with dtype of ndarray. """
    assert isinstance(ndarray, np.ndarray)
    dtype_ = ndarray.dtype
    if dtype_ == np.float64 or dtype_ == np.float32:
        return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
    elif dtype_ == np.int64:
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
    else:
        raise ValueError(("The input should be numpy ndarray. Instaed got %s" % ndarray.dtype))


def convert_to_tf_record(X, Y=None, filename=None, verbose=True):
    """
    Reference: https://gist.github.com/swyoon/8185b3dcf08ec728fb22b99016dd533f
    :param X:
    :param y:
    :param filename:
    :return:
    """

    if filename is not None:
        filename = os.path.join(save_dir, filename + '.tfrecords')
    else:
        filename = os.path.join(save_dir, 'export.tfrecords')

    writer = tf.python_io.TFRecordWriter(filename)

    if verbose:
        print("Serializing %s examples into %s" % (X.shape[0], filename))

    # Enter all features here
    for idx in range(X.shape[0]):

        d_feature = {}

        x = X[idx, 0]
        d_feature['comment_text'] = _bytes_feature(str.encode(x))

        if Y is not None:
            d_feature['toxic'] = _int64_feature(Y[idx, 0])
            d_feature['severe_toxic'] = _int64_feature(Y[idx, 1])
            d_feature['obscene'] = _int64_feature(Y[idx, 2])
            d_feature['threat'] = _int64_feature(Y[idx, 3])
            d_feature['insult'] = _int64_feature(Y[idx, 4])
            d_feature['identity_hate'] = _int64_feature(Y[idx, 5])

        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)

        writer.write(example.SerializeToString())

    writer.close()


if __name__ == "__main__":

    train_df = pd.read_csv("../data/train.csv", encoding="utf8")
    test_df = pd.read_csv("../data/test.csv", encoding="utf8")

    # Convert multiclass labels into single-coded label
    feature_cols = ['comment_text']
    multiclass_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    # train_df_single = convert_multiclass(train_df, feature_cols, multiclass_cols)

    train_X_imb = train_df.loc[:, 'comment_text'].as_matrix().reshape(-1, 1)
    train_y_imb = train_df.loc[:, multiclass_cols].values

    train_X, val_X, train_y, val_y = train_test_split(train_X_imb,
                                                      train_y_imb,
                                                      test_size=0.3,
                                                      random_state=42)
    test_X = test_df.loc[:, feature_cols].as_matrix().astype("U").reshape(-1, 1)

    convert_to_tf_record(train_X, train_y, filename="train")
    convert_to_tf_record(val_X, val_y, filename="val")
    convert_to_tf_record(test_X, filename="test")

    print("Done!")