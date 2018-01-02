import tensorflow as tf
import numpy as np
from distutils.version import LooseVersion
import warnings
import helper


def getTensors(loaded_graph):
    InputTensor = tf.Graph.get_tensor_by_name(loaded_graph,'input:0')
    InitialStateTensor = tf.Graph.get_tensor_by_name(loaded_graph,'initial_state:0')
    FinalStateTensor = tf.Graph.get_tensor_by_name(loaded_graph, 'final_state:0')
    ProbsTensor = tf.Graph.get_tensor_by_name(loaded_graph, 'probs:0')

    return InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor

def pick_word(probabilities, int_to_vocab):
    index = np.random.choice(np.array(len(probabilities)), 1, p=probabilities)
    return int_to_vocab[index[0]]

def genScript():
    _, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
    #print("vocab to int : ",vocab_to_int)
    seq_length, load_dir = helper.load_params()
    gen_length = 200
    prime_word = 'bart_simpson'
    #'bart_simpson'  # 'moe_szyslak'
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph('save.meta')
        loader.restore(sess, load_dir)
        input_text, initial_state, final_state, probs = getTensors(loaded_graph)
        gen_sentences =[prime_word + ':']
        
        prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

        # Generate sentences
        for n in range(gen_length):
            dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
            
            dyn_seq_length = len(dyn_input[0])

            probabilities, prev_state = sess.run(
                [probs, final_state],
                {input_text:dyn_input, initial_state:prev_state}
            )

            pred_word = pick_word(probabilities[0][dyn_seq_length-1], int_to_vocab)

            gen_sentences.append(pred_word)
        
        # remove tokens
        tv_script = ' '.join(gen_sentences)
        for key, token in token_dict.items():
            ending = ' ' if key in ['\n', '(', '"'] else ''
            tv_script = tv_script.replace(' ' + token.lower(), key)
        tv_script = tv_script.replace('\n ', '\n')
        tv_script = tv_script.replace('( ','(')

        print(tv_script)

def main():
    genScript()


if __name__ == '__main__':
    main()


