import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import pandas as pd

HIDDEN_LAYER_DIMENSIONS = 128
NUMBER_LAYERS = 1
SEQUENCE_LENGTH = 32
EPOCHS = 200
BATCH_SIZE = 10
NUM_OUTPUT_CLASSES = 2
TRAIN = False
USE_SAVED_MODEL = True
TRAINING_FILE_PATH = '../data/data.csv'
VOCAB_SIZE_VAR = None  # Size of the model's vocabulary. Depends on the training data
LOG_FREQUENCY = 20
META_GRAPH_PATH = 'saved_model/-' + str(EPOCHS) + '.meta'
CHECKPOINT_DIR = 'saved_model/'


def load_data_and_vocab(file_path):
    '''
    Load data from the file path and create a pandas data frame with char2ind and ind2char dictionaries
    :param file_path: The path to the csv file containing the training data. It should have two columns: text and class
    :return: returns file contents as a data frame and character/index lookup dictionaries
    '''
    rel_path = file_path
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    training_data = pd.read_csv(path, dtype={'class': int})
    all_lines = training_data['text']
    all_chars = []
    for line in all_lines:
        for character in line:
            all_chars.append(character)
    all_chars = sorted(list(set(all_chars)))
    all_chars.append('UNK')  # to take care of characters not seen in training data

    data_size, vocab_size = training_data['text'].shape[0], len(all_chars)
    print('data has %d samples, %d unique characters.' % (data_size, vocab_size))

    char_to_ix = {ch: i for i, ch in enumerate(all_chars)}
    ix_to_char = {i: ch for i, ch in enumerate(all_chars)}

    return training_data, char_to_ix, ix_to_char, data_size, vocab_size


def create_model():
    '''
    Creates the model to train
    :return: Returns the last output of a sequential model using LSTMs along with features and labels placeholders
    '''
    sess = tf.Session()
    input_sequences = tf.placeholder(tf.float32, [None, SEQUENCE_LENGTH, VOCAB_SIZE_VAR], name='input_sequences')
    labels = tf.placeholder(tf.float32, [None, NUM_OUTPUT_CLASSES], name='labels')
    rnn_cell = rnn.MultiRNNCell([rnn.LSTMCell(HIDDEN_LAYER_DIMENSIONS)] * NUMBER_LAYERS)
    outputs, states = tf.nn.static_rnn(cell=rnn_cell,
                                       inputs=tf.unstack(input_sequences, axis=1),
                                       dtype=tf.float32,
                                       )
    predictions = tf.layers.dense(inputs=outputs[-1],
                                  units=NUM_OUTPUT_CLASSES,
                                  name='predictions')
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions,
                                                                     labels=labels),
                                                                     name='cost')
    accuracy = tf.contrib.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=tf.argmax(predictions, 1))
    accuracy = tf.identity(accuracy, name='accuracy')
    optimizer = tf.train.AdamOptimizer(name='optimizer')
    minimizer = optimizer.minimize(cost, name='minimizer')
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(name='saver')
    return sess, input_sequences, labels, minimizer, accuracy, saver


def load_model():
    sess = tf.Session()
    saver = tf.train.import_meta_graph(META_GRAPH_PATH)
    saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_DIR))
    graph = sess.graph
    input_sequences = graph.get_tensor_by_name('input_sequences:0')
    labels = graph.get_tensor_by_name('labels:0')
    minimizer = graph.get_operation_by_name('minimizer')
    accuracy = graph.get_tensor_by_name('accuracy_1:0')

    return sess, input_sequences, labels, minimizer, accuracy, saver


def char2vec(char):
    '''
    Converts a given character to its vector representation
    :param char: A character to convert to its vector representation
    :return: A one-hot representation of the character
    '''
    vector = np.zeros(VOCAB_SIZE_VAR, dtype=np.float32)
    index = char_to_ix.get(char, char_to_ix['UNK'])
    vector[index] = 1
    return vector


def get_random_batch():
    '''
    Gets a randomized batch of sequences from the training data and their respective classes
    :return:
    '''
    random_indexes = np.random.randint(0, data_size, size=BATCH_SIZE)
    batch_inputs = []
    batch_outputs = []
    for index in random_indexes:
        inputs = []
        output = np.zeros(2, dtype=np.float32)
        output[data['class'][index]] = 1
        for i in range(len(data['text'][index])):
            if i < SEQUENCE_LENGTH:
                char_to_input = data['text'][index][i]
                inputs.append(char2vec(char_to_input))
        while len(inputs) < SEQUENCE_LENGTH:
            inputs.append(np.zeros(VOCAB_SIZE_VAR, dtype=np.float32))
        batch_inputs.append(np.array(inputs, dtype=np.float32))
        batch_outputs.append(output)
    batch_inputs = np.array(batch_inputs, dtype=np.float32)
    batch_outputs = np.array(batch_outputs, dtype=np.float32)
    return {input_sequences: batch_inputs, labels: batch_outputs}


data, char_to_ix, ix_to_char, data_size, VOCAB_SIZE_VAR = load_data_and_vocab(TRAINING_FILE_PATH)

if USE_SAVED_MODEL:
    sess, input_sequences, labels, minimizer, accuracy, saver = load_model()
else:
    sess, input_sequences, labels, minimizer, accuracy, saver = create_model()

for e in range(EPOCHS):
    feed_dict = get_random_batch()
    if TRAIN:
        sess.run([minimizer], feed_dict=feed_dict)

    accuracy_results = sess.run([accuracy], feed_dict=feed_dict)[0]

    if e % LOG_FREQUENCY == 0:
        print('Epoch: ' + str(e) + ', Average Classification Error: {:,.0%}'.format(1 - accuracy_results))

if TRAIN:
    saver.save(sess=sess,
               save_path=CHECKPOINT_DIR,
               global_step=EPOCHS)

sess.close()
