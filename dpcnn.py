from keras import Model
from keras.layers import Dense, Embedding, Input, Flatten, Concatenate,\
    SpatialDropout1D, Conv1D, MaxPooling1D, Add, Reshape, ZeroPadding1D
from keras.preprocessing.sequence import pad_sequences
from keras_callbacks import *
import numpy as np
from metrics import f1
import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
from keras.losses import categorical_crossentropy
from sklearn.metrics import f1_score
from torch_data import UserCommentDataset, calculate_labels, save_predictions
from train_config import data_path_config, targets
from gensim.models import KeyedVectors
from utils import *
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))
K.clear_session()


def dp_block(input):
    pool = ZeroPadding1D((0, 1))(input)
    pool = MaxPooling1D(pool_size=3, strides=2, padding='valid')(pool)
    conv = Conv1D(filters=250, strides=1, kernel_size=3, padding='same')(pool)  # keep dimension the same
    conv = Conv1D(filters=250, strides=1, kernel_size=3, padding='same', use_bias=False)(conv)
    shortcut = Add()([conv, pool])
    return shortcut


def decompose_targets(data):
    decomposed_list = []
    for idx, _ in enumerate(targets):
        target_list = [x[idx] for x in data]
        target_array = np.vstack(target_list)
        decomposed_list.append(target_array)
    return decomposed_list


def build_model(embedding_layer, maxlen, targets):
    sent_input = Input((maxlen,), dtype='int32')
    sent_embedded = embedding_layer(sent_input)

    # region embedding
    conv = Conv1D(filters=250, strides=1, kernel_size=3, padding='same')(sent_embedded)

    while conv.shape[1] > 2:
        conv = dp_block(conv)

    output_list = []
    for target in targets:
        flat_target = Flatten(name='flatten' + target)(conv)
        output_target = Dense(4, name='output' + target, activation='softmax')(flat_target)
        output_list.append(output_target)
    model = Model(sent_input, output_list)
    return model


def main():
    maxlen = 1024
    train_dataset = UserCommentDataset(data_path_config['train_data_path'],
                                       targets=targets,
                                       content='jieba_seg',
                                       transformers=None)
    validate_dataset = UserCommentDataset(data_path_config['valid_data_path'],
                                          targets=targets,
                                          content='jieba_seg',
                                          transformers=None)
    test_dataset = UserCommentDataset(data_path_config['test_data_path'],
                                      targets=None,
                                      content='jieba_seg',
                                      transformers=None)
    train_sentences, train_targets = zip(*train_dataset)
    validate_sentences, validate_targets = zip(*validate_dataset)
    test_sentences, _ = zip(*test_dataset)
    Y_train = decompose_targets(train_targets)
    Y_valid = decompose_targets(validate_targets)
    w2v = KeyedVectors.load_word2vec_format(data_path_config['embedding_path'],
                                            binary=True, unicode_errors='ignore')
    tok2idx = build_tok2idx(w2v)
    w2v_matrix = build_embedding_matrix(tok2idx, w2v, data_path_config['embedding_dim'])
    X_train_seq = texts_to_sequences(train_sentences, tok2idx)
    X_valid_seq = texts_to_sequences(validate_sentences, tok2idx)
    X_test_seq = texts_to_sequences(test_sentences, tok2idx)
    embedding_layer = Embedding(input_dim=w2v_matrix.shape[0],
                                output_dim=w2v_matrix.shape[1],
                                weights=[w2v_matrix],
                                trainable=False)
    model = build_model(embedding_layer, maxlen, targets)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(X_train_pad, Y_train)


if __name__ == '__main__':
    main()