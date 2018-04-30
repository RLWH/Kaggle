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


def make_cell(state_dim):
    lstm_initializer = tf.random_uniform_initializer(-0.1, 0.1)
    return tf.contrib.rnn.LSTMCell(state_dim, initializer=lstm_initializer)


def make_multi_cell(state_dim, num_layers):
    cells = [make_cell(state_dim) for _ in range(num_layers)]
    return tf.contrib.rnn.MultiRNNCell(cells)


class Model:

    def __init__(self, input_num_vocab):

        # Define hyperparameters
        self.NUM_EPOCHS = 300
        self.RNN_STATE_DIM = 512
        self.RNN_NUM_LAYERS = 2
        self.ENCODER_EMBEDDING_DIM = 64


        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.0003

        self.INPUT_NUM_VOCAB = input_num_vocab
        self.DROPOUT_RATE = 0.5

    def inference(self, inputs, training=False):
        """
        Build the model.
        :param features: Features from the data
        :return: Logits
        """

        # Remarks - Input is with shape [None, seq_size, input_dim]
        # encoder_seq_length = tf.shape(inputs).eval()[1]
        embedding_dim = 300

        with tf.device('/cpu:0'):
            with tf.variable_scope("embedding"):
                # Embedding matrix
                embedding = tf.get_variable("embedding_weight", shape=[self.INPUT_NUM_VOCAB + 1, embedding_dim])

                # Embedding lookup
                embedded_input = tf.nn.embedding_lookup(embedding, inputs)

        with tf.variable_scope("first_cell") as scope:
            # Create LSTM Cell
            cell = make_multi_cell(self.RNN_STATE_DIM, self.RNN_NUM_LAYERS)

            # Runs the cell on the input to obtain tensor for putputs and states
            outputs, states = tf.nn.dynamic_rnn(cell,
                                                embedded_input,
                                                dtype=tf.float32)

        with tf.variable_scope("combine"):
            # flatten = tf.reshape(outputs, [-1, self.RNN_STATE_DIM])
            output_t = tf.transpose(outputs, [1, 0, 2])
            print(output_t.get_shape())
            last = tf.gather(output_t, tf.cast((tf.shape(output_t)[0]), dtype=tf.int32) - 1)

        with tf.variable_scope("dense1"):
            dense1 = tf.layers.dense(last,  units=192, activation=tf.nn.relu, name="dense1")

        with tf.variable_scope("dropout1"):
            dropout1 = tf.layers.dropout(dense1, rate=self.DROPOUT_RATE, name="dropout1", training=training)

        with tf.variable_scope("dense2"):
            dense2 = tf.layers.dense(dropout1, units=64, activation=tf.nn.relu)

        with tf.variable_scope("pred"):
            logits = tf.layers.dense(inputs=dense2, units=2)

        return logits

    def loss(self, logits, labels):
        """
        Loss function
        :param logits:
        :param labels:
        :return:
        """
        # Assume labels are one_hot encoded
        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(labels, depth=2), logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def train(self, total_loss, global_step):
        """
        Train a model
        Create an optimizer and apply to all trainable variables. Add moving average for all trainable variables
        :return: train_op
        """
        train_op = tf.train.AdagradOptimizer(learning_rate=self.LEARNING_RATE).minimize(total_loss,
                                                                                        global_step=global_step)

        return train_op