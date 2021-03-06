"""
Evaluation script
"""

import tensorflow as tf
import data_input
import time

from config import config
from model import Model
from utils.word_processing import read_vocab


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'logs/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'train_eval',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval. """)
tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Batch size of each eval """)
tf.app.flags.DEFINE_string('checkpoint_dir', '/checkpoint',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

def evaluate_once(graph, iterator, auc, acc, auc_update_op, accuracy_update_op, summary_op, saver, summary_writer):

    with tf.Session(graph=graph) as sess:

        sess.run(tf.local_variables_initializer())
        sess.run(iterator.initializer)
        sess.run(tf.tables_initializer())

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

        sess.run([auc_update_op, accuracy_update_op])
        auc_val, acc_val, summary = sess.run([auc, acc, summary_op])
        summary_writer.add_summary(summary, global_step)

        print("Eval step %s: Accuracy: %s, AUC: %s" % (global_step, acc_val, auc_val))


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

        eval_input = data_input.train_eval_input_fn(eval_dataset, table, batch_size=FLAGS.batch_size)

        # Transform the dataset into tf.data.Dataset. Build iterator
        iterator = eval_input.make_initializable_iterator()
        features, labels = iterator.get_next()

        # Infer the logits and loss
        logits, prediction = model.inference(features, training=False)
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

        # Initiate a saver and pass in all saveable variables
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

        while True:
            evaluate_once(g, iterator, auc, accuracy, accuracy_update_op, auc_update_op,
                          summary_op, saver, summary_writer)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv):
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    evaluate()


if __name__ == "__main__":
    tf.app.run()
