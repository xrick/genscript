import helper
import os
import pickle
import tensorflow as tf
from tensorflow.contrib import seq2seq
from distutils.version import LooseVersion
import warnings
import problem_unittests as tests
import numpy as np

assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))
'''
shared constants
'''
data_dir = './data/simpsons/moes_tavern_lines.txt'
num_epochs = 80
batch_size = 192
rnn_size = 192
embed_dim = 256
seq_length = 20
learning_rate = 0.02
show_every_n_batches = 10
save_dir = "./save"
#load the simpsons script
def loadData():
    text = helper.load_data(data_dir)
    return text

def create_lookup_tables(text):
    vocab = set(text)
    vocab_to_int = {c : i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    return vocab_to_int, int_to_vocab

#Tokenize Punctuation
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    在此token中，並未把 ":"拿掉，因為保留":"做為劇本角色的名字的一部份，在訓練習完後，以":"做為尋找角色的關
    鍵字元。
    """
    tokenDict = {
        '.':'<period>',
        ',':'<comma>',
        '"':'<quotation_mark>',
        ';':'<semicolon>',
        '!':'<exclamation_mark>',
        '?':'<question_mark>',
        '(':'<left_parenthesis>',
        ')':'<right_parenthesis>',
        '--':'<dash>',
        '\n':'<renturn>'
    }
    return tokenDict

'''
setup tensorflow input
'''
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    Input = tf.placeholder(tf.int32, (None,None), name='input')
    targets = tf.placeholder(tf.int32, (None,None), name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    return (Input, targets, learning_rate)

'''
Build Rnn Cell and Initialize
'''
def get_init_cell(batch_size, rnn_size):
    layers = 1
    cells = []
    for i in range(layers):
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        cells.append(lstm)
    Cell = tf.contrib.rnn.MultiRNNCell(cells)
    InitialState = tf.identity(Cell.zero_state(batch_size, tf.float32), name='initial_state')
    return Cell, InitialState

'''
Word Embedding
'''
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_normal([vocab_size,embed_dim], stddev=0.05))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed


def get_batches(int_text, batch_size, seq_length):
    print("seq_length",seq_length)
    characters_per_batch = seq_length * batch_size
    n_batches = (len(int_text)-1) // characters_per_batch
    xData = np.array(int_text[: n_batches * characters_per_batch])
    yData = np.array(int_text[1: n_batches * characters_per_batch] + [0])
    x_batches = np.split(xData.reshape(batch_size, -1), n_batches, 1)
    y_batches = np.split(yData.reshape(batch_size, -1), n_batches, 1)
    batches = np.array(list(zip(x_batches,y_batches)))
    return batches


'''
Build RNN
'''
def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, state = tf.nn.dynamic_rnn(cell,inputs, dtype=tf.float32)
    FinalState = tf.identity(state, name='final_state')
    return outputs, FinalState
    '''
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    FinalState = tf.identity(state, name='final_state')
    return outputs, FinalState
    '''
    


"""
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
"""
def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    embed = get_embed(input_data, vocab_size, rnn_size)
    x, FinalState = build_rnn(cell, embed)
    Logits = tf.contrib.layers.fully_connected(x, vocab_size, activation_fn=None)
    return Logits, FinalState

"""
preprocess methods
"""


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()
    return data

def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)

    # Ignore notice, since we don't use it for analysing the data
    text = text[81:]
    #print("text[81:] is :",text)
    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab,
                 token_dict), open('preprocess.p', 'wb'))

def main():
    
    #tests.test_build_nn(build_nn)
    preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
    int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
    #build graph to run the train
    train_graph = tf.Graph()
    with train_graph.as_default():
        vocab_size = len(int_to_vocab)
        input_text, targets, lr = get_inputs()
        input_data_shape = tf.shape(input_text)
        cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
        print("cell is: ",cell)
        print("rnn_size is: ",rnn_size)
        print("input_text is: ",input_text)
        print("vocab_size is: ",vocab_size)
        print("input_data_shape is: ",input_data_shape)
        print("embed_dim is: ",embed_dim)
        logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)
        print("logits is: ",logits)
        # Probabilities for generating words
        probs = tf.nn.softmax(logits, name='probs')
        print("probs is: ",probs)
        # Loss function
        cost = seq2seq.sequence_loss(logits,targets,tf.ones([input_data_shape[0], input_data_shape[1]]))
        print("cost is: ",cost)
        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping (google it)
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
    
    batches = get_batches(int_text, batch_size, seq_length)

    with tf.Session(graph = train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(num_epochs):
            state = sess.run(initial_state, {input_text: batches[0][0]})

            for batch_i, (x, y) in enumerate(batches):
                feed = {
                    input_text : x,
                    targets: y,
                    initial_state: state,
                    lr: learning_rate
                }
                train_loss, state, _ = sess.run([cost,final_state,train_op],feed)

                if(epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                    print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        epoch_i,
                        batch_i,
                        len(batches),
                        train_loss))
        # save model
        saver = tf.train.Saver()
        saver.save(sess, save_dir)
        print('Model Trained and Saved')

    
    
if __name__ == '__main__':
    main()
