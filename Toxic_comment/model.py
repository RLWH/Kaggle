"""
Data Model main script

Summary of available functions:

# Preprocessing function (Not available in this model)
inputs, labels = preprocessing()

# Compute inference on the model inputs to make a prediction
predictions = inference(inputs)

# Compuate the total loss of the prediction w.r.t. to the labels
loss = loss(predictions, labels)

# Create a graph to run one step of training w.r.t. the loss
train_op = train(loss, global_step)

"""

import numpy as np
import pandas as pd
import tensorflow as tf
import data_input
import re

from config import config


def make_cell(state_dim):
    lstm_initializer = tf.random_uniform_initializer(-0.1, 0.1)
    return tf.contrib.rnn.LSTMCell(state_dim, initializer=lstm_initializer)


def make_multi_cell(state_dim, num_layers):
    cells = [make_cell(state_dim) for _ in range(num_layers)]
    return tf.contrib.rnn.MultiRNNCell(cells)


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % "Tower", '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


class Model:

    def __init__(self, input_num_vocab):

        # Define hyperparameters
        self.RNN_STATE_DIM = config.RNN_STATE_DIM
        self.RNN_NUM_LAYERS = config.RNN_NUM_LAYERS
        self.ENCODER_EMBEDDING_DIM = config.ENCODER_EMBEDDING_DIM
        self.num_class = config.NUM_CLASS
        self.LEARNING_RATE = config.LEARNING_RATE

        # Dense layer units
        self.DENSE1_UNIT = config.DENSE1_UNIT
        self.DENSE2_UNIT = config.DENSE2_UNIT

        self.INPUT_NUM_VOCAB = input_num_vocab
        self.DROPOUT_RATE = config.DROPOUT_RATE

    def inference(self, inputs, training=False):
        """
        Build the model.
        :param features: Features from the data
        :return: Logits
        """

        # Remarks - Input is with shape [None, seq_size, input_dim]
        # encoder_seq_length = tf.shape(inputs).eval()[1]
        embedding_dim = config.ENCODER_EMBEDDING_DIM

        with tf.device('/cpu:0'):
            with tf.variable_scope("embedding"):
                # Embedding matrix
                embedding = tf.get_variable("embedding_weight", shape=[self.INPUT_NUM_VOCAB + 1, embedding_dim])

                # Embedding lookup
                embedded_input = tf.nn.embedding_lookup(embedding, inputs)

        with tf.variable_scope("first_cell") as scope:
            # Create LSTM Cell
            cell = make_multi_cell(self.RNN_STATE_DIM, self.RNN_NUM_LAYERS)

            # Runs the cell on the input to obtain tensor for outputs and states
            outputs, states = tf.nn.dynamic_rnn(cell,
                                                embedded_input,
                                                dtype=tf.float32)

        with tf.variable_scope("combine"):
            # flatten = tf.reshape(outputs, [-1, self.RNN_STATE_DIM])
            output_t = tf.transpose(outputs, [1, 0, 2])
            # print(output_t.get_shape())
            last = tf.gather(output_t, tf.cast((tf.shape(output_t)[0]), dtype=tf.int32) - 1)

        with tf.variable_scope("dense1"):
            dense1 = tf.layers.dense(last,  units=self.DENSE1_UNIT, activation=tf.nn.relu, name="dense1")

        with tf.variable_scope("dropout1"):
            dropout1 = tf.layers.dropout(dense1, rate=self.DROPOUT_RATE, name="dropout1", training=training)

        with tf.variable_scope("dense2"):
            dense2 = tf.layers.dense(dropout1, units=self.DENSE2_UNIT, activation=tf.nn.relu)

        with tf.variable_scope("pred"):
            logits = tf.layers.dense(inputs=dense2, units=self.num_class)

        return logits

    def loss(self, logits, labels):
        """
        Loss function
        :param logits:
        :param labels:
        :return:
        """
        # Assume labels are one_hot encoded
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits,
                                                       name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        prediction_op = tf.greater(logits, config.PRED_THRESHOLD)

        correct_prediction_op = tf.equal(prediction_op, tf.round(labels))
        mean_accuracy = tf.reduce_mean(tf.cast(correct_prediction_op, tf.float32))
        tf.add_to_collection('mean_acc', mean_accuracy)

        return tf.add_n(tf.get_collection('losses'), name='total_loss'), \
               tf.reduce_mean(tf.get_collection('mean_acc'), reduction_indices=0, name='total_mean_acc')

    def train(self, total_loss, global_step):
        """
        Train a model
        Create an optimizer and apply to all trainable variables. Add moving average for all trainable variables
        :return: train_op
        """
        train_op = tf.train.AdagradOptimizer(learning_rate=self.LEARNING_RATE).minimize(total_loss,
                                                                                        global_step=global_step)

        return train_op
