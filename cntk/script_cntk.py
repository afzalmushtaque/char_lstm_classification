from cntk.ops import sequence
import os
import pandas as pd
from cntk import Trainer, load_model, input_variable
from cntk.layers import LSTM, Recurrence, Dense, For, Sequential
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.learners import momentum_schedule_per_sample, adam
import numpy as np


HIDDEN_LAYER_DIMENSIONS = 128
NUMBER_LAYERS = 1
SEQUENCE_LENGTH = 32
EPOCHS = 2000
BATCH_SIZE = 10
NUM_OUTPUT_CLASSES = 2
TRAIN = False
USE_SAVED_MODEL = True
TRAINING_FILE_PATH = '../data/data.csv'
MODEL_FILE_PATH = 'saved_model.dnn'
VOCAB_SIZE_VAR = None  # Size of the model's vocabulary. Depends on the training data
LOG_FREQUENCY = 100


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


def create_model_placeholders():
    '''
    Create placeholders for getting features and labels into the model
    :return: returns a feature placeholder and a label placeholder
    '''
    input_sequence = sequence.input_variable(shape=VOCAB_SIZE_VAR, name='input_sequence')
    label_sequence = input_variable(shape=NUM_OUTPUT_CLASSES, name='label_sequence')
    return input_sequence, label_sequence


def create_model():
    '''
    Creates the model to train
    :return: Returns the last output of a sequential model using LSTMs
    '''
    return Sequential([
        For(range(NUMBER_LAYERS), lambda: Sequential([Recurrence(LSTM(HIDDEN_LAYER_DIMENSIONS))])),
        sequence.last,
        Dense(NUM_OUTPUT_CLASSES)
    ])


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
            char_to_input = data['text'][index][i]
            inputs.append(char2vec(char_to_input))
        batch_inputs.append(np.array(inputs, dtype=np.float32))
        batch_outputs.append(output)
    batch_outputs = np.array(batch_outputs, dtype=np.float32)
    return {input_sequences: batch_inputs, labels: batch_outputs}


data, char_to_ix, ix_to_char, data_size, VOCAB_SIZE_VAR = load_data_and_vocab(TRAINING_FILE_PATH)
input_sequences, labels = create_model_placeholders()

if USE_SAVED_MODEL:
    model = load_model(MODEL_FILE_PATH)
else:
    model = create_model()

z = model(input_sequences)

ce = cross_entropy_with_softmax(z, labels)
errs = classification_error(z, labels)

momentum_schedule = momentum_schedule_per_sample(0.9990913221888589)
clipping_threshold_per_sample = 5.0
gradient_clipping_with_truncation = True
learner = adam(z.parameters, 0.001, momentum_schedule)
trainer = Trainer(z, (ce, errs), [learner])

for e in range(EPOCHS):
    arguments = get_random_batch()
    if TRAIN:
        trainer.train_minibatch(arguments)
    if e % LOG_FREQUENCY == 0:
        print('Epoch: ' + str(e) + ', Average Classification Error: {:,.0%}'.format(
            pd.DataFrame(errs.eval(arguments))[0].mean()))

if TRAIN:
    z.save(MODEL_FILE_PATH)
    print("Saved model to '%s'" % MODEL_FILE_PATH)
