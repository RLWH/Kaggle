"""
Evaluation script
"""

import tensorflow as tf
import numpy as np
import data_input

from model import Model


def evaluate(dataset, all_symbols):
    """
    Eval the model for a number of steps
    :return:
    """

    with tf.Graph().as_default() as g:
        # Init the model
        model = Model(input_num_vocab=len(all_symbols))

        # Import dataset
        eval_dataset = data_input.read_data("data/val.tfrecords")

        table = tf.contrib.lookup.index_table_from_tensor(
            mapping=tf.constant(all_symbols), num_oov_buckets=1, default_value=-1)

        eval_input = data_input.train_eval_input_fn(eval_dataset, table, batch_size=5)

        # Transform the dataset into tf.data.Dataset. Build iterator
        iterator = dataset.make_one_shot_generator()
        features, labels = iterator.get_next()

        # Infer the logits and loss
        logits = model.inference(features)

        # Calculate loss

        with tf.Session() as sess:

            while True:
                try:
                    sess.run(loss)
                except tf.errors.OutOfRangeError:
                    break


def main():
    evaluate()


if __name__ == "__main__":
    tf.app.run()
