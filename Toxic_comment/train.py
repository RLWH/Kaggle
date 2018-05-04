"""
Training operation
"""
import tensorflow as tf
import time
import data_input
import csv

from model import Model
from config import config
from utils.word_processing import read_vocab, build_vocab
from data_input import read_data, train_eval_input_fn


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'logs/train',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('max_steps', 10,
                            """Maximum steps of the run""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 1,
                            """How often to log results to the console.""")

def train(dataset, all_symbols):
    """
    Train the model for a number of steps
    :return:
    """

    global_step = tf.train.get_or_create_global_step()

    # Init the model
    model = Model(input_num_vocab=len(all_symbols))

    # Transform the dataset into tf.data.Dataset. Build iterator
    iterator = dataset.make_initializable_iterator()
    features, labels = iterator.get_next()

    # Infer the logits and loss
    logits, prediction = model.inference(features, training=True)

    # Calculate loss
    loss = model.loss(logits, labels)

    # Calculating the accuracy and auc
    correct_prediction = tf.greater(prediction, config.PRED_THRESHOLD)
    accuracy, accuracy_update_op = tf.metrics.accuracy(labels, tf.cast(correct_prediction, tf.int64))
    auc, auc_update_op = tf.metrics.auc(labels, prediction)

    mean_accuracy = tf.reduce_mean(accuracy)
    mean_auc = tf.reduce_mean(auc)

    # Add to tensorboard
    loss_hist = tf.summary.scalar('total_loss', loss)
    acc_hist = tf.summary.scalar('mean_acc', mean_accuracy)
    auc_hist = tf.summary.scalar('mean_auc', mean_auc)

    summary_op = tf.summary.merge_all()

    # Build a Graph that trains the model with one batch of example and update the model params
    train_op = model.train(loss, global_step)

    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        iterator.initializer,
                                                        tf.tables_initializer()))

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.train.MonitoredTrainingSession(
            scaffold=scaffold,
            checkpoint_dir=FLAGS.checkpoint_dir,
            save_summaries_steps=FLAGS.log_frequency,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(loss),
                   tf.train.LoggingTensorHook({"loss": loss, "acc": mean_accuracy, "auc": mean_auc}, every_n_iter=FLAGS.log_frequency),
                   tf.train.SummarySaverHook(save_steps=FLAGS.log_frequency, output_dir=FLAGS.train_dir,
                                             summary_op=summary_op)],
            config=tf.ConfigProto(log_device_placement=False)) as mon_sess:

        while not mon_sess.should_stop():
            mon_sess.run([train_op, accuracy_update_op, auc_update_op])


def main(argv):

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    # Import existing vocabs
    try:
        all_symbols = read_vocab(source_path='data/vocab.csv')
    except FileNotFoundError:
        print("No vocab file found. Creating one. ")
        all_symbols = build_vocab(source_path='data/train.csv', export_path='data/vocab.csv')

    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(all_symbols), num_oov_buckets=1, default_value=-1)

    train_dataset = data_input.read_data("data/train.tfrecords").repeat(config.NUM_EPOCHS)
    train_input = data_input.train_eval_input_fn(train_dataset, table, batch_size=config.BATCH_SIZE)

    train(train_input, all_symbols)


if __name__ == '__main__':
    tf.app.run(main)