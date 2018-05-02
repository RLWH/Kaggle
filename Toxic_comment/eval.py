"""
Evaluation script
"""

import tensorflow as tf
import numpy as np
import data_input
import csv

from config import config
from model import Model
from utils.word_processing import read_vocab


def evaluate_once(saver, summary_writer):

    with tf.Session() as sess:
        # Read checkpoint if checkpoint exists
        ckpt = tf.train.get_checkpoint_state("checkpoint")

        if ckpt and ckpt.model_checkpoint_path:
            # Restore the checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0, --> Global step = Last digit
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            # If no checkpoint found, return
            print("No checkpoint found")
            return




def evaluate():
    """
    Eval the model for a number of steps
    :return:
    """

    with tf.Graph().as_default() as g:

        # Read the vocabularies
        all_symbols = read_vocab('data/vocab.csv')

        # Init the model
        model = Model(input_num_vocab=len(all_symbols))

        # Import dataset
        eval_dataset = data_input.read_data("data/val.tfrecords")

        table = tf.contrib.lookup.index_table_from_tensor(
            mapping=tf.constant(all_symbols), num_oov_buckets=1, default_value=-1)

        eval_input = data_input.train_eval_input_fn(eval_dataset, table, batch_size=5)

        # Transform the dataset into tf.data.Dataset. Build iterator
        iterator = eval_dataset.make_one_shot_generator()
        features, labels = iterator.get_next()

        # Infer the logits and loss
        logits = model.inference(features)

        # Calculate predictions
        prediction_op = tf.greater(logits, config.PRED_THRESHOLD)

        mean_accuracy_op = tf.equal(prediction_op, tf.round(labels))

        # Initiate a saver and pass in all saveable variables
        saver = tf.train.Saver()


def main():
    evaluate()


if __name__ == "__main__":
    tf.app.run()
