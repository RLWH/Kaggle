"""
A testing program to validate tfrecord
"""
import os
import tensorflow as tf

save_dir = "../data"

if __name__ == '__main__':

    filename = os.path.join(save_dir, 'train.tfrecords')

    # Use tf.python_io.tf_record_iterator to read records from a TFRecords file
    record_iterator = tf.python_io.tf_record_iterator(filename)
    seralized_example = next(record_iterator)

    example = tf.train.Example()
    example.ParseFromString(seralized_example)

    comment_text = example

    # print(comment_text)

    with tf.Session() as sess:

        feature = {
            'comment_text': tf.FixedLenFeature((), dtype=tf.string, default_value=""),
            'label': tf.FixedLenFeature((), dtype=tf.int64, default_value=0)
        }

        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)

        comment_text = tf.cast(features['comment_text'], tf.string)
        label = tf.cast(features['label'], tf.int64)

        comment_texts, labels = tf.train.shuffle_batch([comment_text, label], batch_size=10, capacity=30, num_threads=1,
                                                min_after_dequeue=10)

        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print(sess.run(comment_texts))

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()





