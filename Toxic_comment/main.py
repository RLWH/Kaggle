"""
Main execution script
"""

import tensorflow as tf
import numpy as np
import model
import train
import eval
import data_input


def main(argy):

    # Build vocabularies
    all_symbols = data_input.build_vocab()

    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(all_symbols), num_oov_buckets=1, default_value=-1)

    # Test using tfRecords
    tfrecord_paths = {'train': 'data/train.tfrecords',
                      'val': 'data/val.tfrecords',
                      'test': 'data/test.tfrecords'}

    print("Test dataset API with tfrecords")

    train_dataset, val_dataset, test_dataset = data_input.read_data(tfrecord_paths)

    train_input_dataset = data_input.train_eval_input_fn(train_dataset, table, batch_size=5)
    eval_input_dataset = data_input.train_eval_input_fn(val_dataset, table, batch_size=5)

    iterator = train_input_dataset.make_initializable_iterator()

    # train.train(train_input_dataset, all_symbols)
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)
        features, labels = iterator.get_next()

        print(sess.run(features))
        print(sess.run(tf.shape(features)))
        print(sess.run(labels))
        print(sess.run(tf.shape(labels)))


if __name__ == '__main__':
    # FLAGS = parser.parse_args()
    tf.app.run(main)
