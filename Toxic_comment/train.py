"""
Training operation
"""
import tensorflow as tf
import time
import data_input
import csv

from model import Model
from config import config
from utils.word_processing import read_vocab
from data_input import read_data, train_eval_input_fn
from datetime import datetime


def train(dataset, all_symbols):
    """
    Train the model for a number of steps
    :return:
    """

    TRAIN_STEP = 10

    global_step = tf.train.get_or_create_global_step()

    # Init the model
    model = Model(input_num_vocab=len(all_symbols))

    # Transform the dataset into tf.data.Dataset. Build iterator
    iterator = dataset.make_initializable_iterator()
    features, labels = iterator.get_next()

    # Infer the logits and loss
    logits = model.inference(features)

    # Calculate loss
    loss, acc = model.loss(logits, labels)

    # Show accuracy
    # acc = tf.get_collection('mean_acc')

    # Build a Graph that trains the model with one batch of example and update the model params
    train_op = model.train(loss, global_step)

    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        iterator.initializer,
                                                        tf.tables_initializer()))

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.train.MonitoredTrainingSession(
            scaffold=scaffold,
            checkpoint_dir='checkpoint',
            save_summaries_steps=10,
            hooks=[tf.train.StopAtStepHook(last_step=TRAIN_STEP),
                   tf.train.NanTensorHook(loss),
                   tf.train.LoggingTensorHook({"loss": loss, "acc": acc}, every_n_iter=1)],
            config=tf.ConfigProto(log_device_placement=False)) as mon_sess:

        while not mon_sess.should_stop():
            mon_sess.run(train_op)


def main(argv):
    # Import existing vocabs
    all_symbols = read_vocab(source_path='data/vocab.csv')

    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(all_symbols), num_oov_buckets=1, default_value=-1)

    train_dataset = data_input.read_data("data/train.tfrecords")
    train_input = data_input.train_eval_input_fn(train_dataset, table, batch_size=config.BATCH_SIZE)

    train(train_input, all_symbols)


if __name__ == '__main__':
    tf.app.run(main)