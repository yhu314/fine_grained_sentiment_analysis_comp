from transformers import *
from train_config import data_path_config, targets, hierarchical_model_config
from torch_data import UserCommentDataset
from gensim.models import KeyedVectors
from utils import build_tok2idx, build_embedding_matrix
import numpy as np
from keras.layers import Input, TimeDistributed, CuDNNGRU, Embedding, Bidirectional,\
    Dense, Reshape, RepeatVector, Permute, Multiply, BatchNormalization, Concatenate
from keras import Model
from keras_callbacks import *
import json


def load_dataset():
    transformers = [remove_special_tokens,
                    process_error_seg,
                    split_sentence,
                    process_sentence,
                    remove_dups]
    train_dataset = UserCommentDataset(data_path_config['train_data_path'],
                                       targets=targets,
                                       content='jieba_seg',
                                       transformers=transformers)
    validate_dataset = UserCommentDataset(data_path_config['valid_data_path'],
                                          targets=targets,
                                          content='jieba_seg',
                                          transformers=transformers)
    test_dataset = UserCommentDataset(data_path_config['test_data_path'],
                                      targets=None,
                                      content='jieba_seg',
                                      transformers=transformers)
    return train_dataset, validate_dataset, test_dataset


def sentence2array(data, tok2idx):
    max_sentences = hierarchical_model_config['max_sentences']
    max_words = hierarchical_model_config['max_sentence_length']
    n_sentences = len(data)
    sentence_array = np.zeros((n_sentences, max_sentences, max_words), dtype='int32')
    for row_idx, sentences in enumerate(data):
        for sent_idx, sentence in enumerate(sentences):
            if sent_idx >= max_sentences:
                continue
            words_list = sentence.split(' ')
            for word_idx, word in enumerate(words_list):
                if word_idx >= max_words:
                    continue
                if word not in tok2idx:
                    continue
                sentence_array[row_idx, sent_idx, word_idx] = tok2idx[word]
    return sentence_array


def self_attention_block(sequence, seq_len):
    att = Dense(1, activation='sigmoid')(sequence)
    weight = Reshape((-1,))(att)
    weight = RepeatVector(seq_len)(weight)
    weight = Permute((2, 1))(weight)
    seq_att = Multiply()([sequence, weight])
    seq_att = BatchNormalization()(seq_att)
    return seq_att


def build_model(config, cache):
    embedding_matrix, max_sentences, max_words = cache
    gru_cells = config['gru_cells']
    output_shape = config['output_shape']
    n_words, embedding_dim = embedding_matrix.shape
    embedding_layer = Embedding(input_dim=n_words,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                trainable=False)

    # Input single sentence
    sentence_input = Input((max_words,), dtype='int32')
    embedded_sentence = embedding_layer(sentence_input)
    sentence_lstm = Bidirectional(CuDNNGRU(gru_cells, return_sequences=False))(embedded_sentence)
    sent_encoder = Model(sentence_input, sentence_lstm)

    # Paragraph level
    review_input = Input(shape=(max_sentences, max_words), dtype='int32')
    review_encoder = TimeDistributed(sent_encoder)(review_input)
    l_lstm_sent = Bidirectional(CuDNNGRU(gru_cells, return_sequences=True))(review_encoder)
    predicts = list()
    for target in targets:
        output_lstm = Bidirectional(CuDNNGRU(gru_cells, return_sequences=True))(l_lstm_sent)
        output = Dense(output_shape, activation='softmax', name=target)(output_lstm)
        predicts.append(output)
    model = Model(review_input, predicts)
    return model


def build_self_attention_model(config, cache):
    embedding_matrix, max_sentences, max_words = cache
    gru_cells = config['gru_cells']
    output_shape = config['output_shape']
    n_words, embedding_dim = embedding_matrix.shape
    embedding_layer = Embedding(input_dim=n_words,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                trainable=False)

    # Input single sentence
    sentence_input = Input((max_words,), dtype='int32')
    embedded_sentence = embedding_layer(sentence_input)
    sentence_lstm = Bidirectional(CuDNNGRU(gru_cells, return_sequences=False))(embedded_sentence)
    sent_encoder = Model(sentence_input, sentence_lstm)

    # Paragraph level
    review_input = Input(shape=(max_sentences, max_words), dtype='int32')
    review_encoder = TimeDistributed(sent_encoder)(review_input)
    l_lstm_sent = Bidirectional(CuDNNGRU(gru_cells, return_sequences=True))(review_encoder)
    predicts = list()
    for target in targets:
        att = Dense(1, activation='sigmoid')(l_lstm_sent)
        weight = Reshape((-1,))(att)
        weight = RepeatVector(gru_cells*2)(weight)
        weight = Permute((2, 1))(weight)
        l_lstm_att = Multiply()([l_lstm_sent, weight])
        l_lstm_att = BatchNormalization()(l_lstm_att)
        l_lstm = Bidirectional(CuDNNGRU(100, return_sequences=False))(l_lstm_att)
        output = Dense(output_shape, activation='softmax', name=target)(l_lstm)
        predicts.append(output)
    model = Model(review_input, predicts)
    return model


def build_self_attention_model_output_update(config, cache):
    embedding_matrix, max_sentences, max_words = cache
    gru_cells = config['gru_cells']
    output_shape = config['output_shape']
    n_words, embedding_dim = embedding_matrix.shape
    embedding_layer = Embedding(input_dim=n_words,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                trainable=True)

    # Input single sentence
    sentence_input = Input((max_words,), dtype='int32')
    embedded_sentence = embedding_layer(sentence_input)
    sentence_lstm = Bidirectional(CuDNNGRU(gru_cells, return_sequences=False))(embedded_sentence)
    sent_encoder = Model(sentence_input, sentence_lstm)

    # Paragraph level
    review_input = Input(shape=(max_sentences, max_words), dtype='int32')
    review_encoder = TimeDistributed(sent_encoder)(review_input)
    l_lstm_sent = Bidirectional(CuDNNGRU(gru_cells, return_sequences=True))(review_encoder)
    predicts = list()
    for target in targets:
        att = Dense(1, activation='sigmoid')(l_lstm_sent)
        weight = Reshape((-1,))(att)
        weight = RepeatVector(gru_cells*2)(weight)
        weight = Permute((2, 1))(weight)
        l_lstm_att = Multiply()([l_lstm_sent, weight])
        l_lstm_att = BatchNormalization()(l_lstm_att)
        l_lstm = Bidirectional(CuDNNGRU(100, return_sequences=False))(l_lstm_att)
        target_mentioned = Dense(1, activation='softmax', name=target+'mentioned')(l_lstm)
        target_sentiment = Dense(output_shape-1, activation='softmax', name=target+'sentiment')(l_lstm)
        target_concat = Concatenate(axis=-1)([target_mentioned, target_sentiment])
        output = Dense(output_shape, activation='softmax', name=target)(target_concat)
        predicts.append(output)
    model = Model(review_input, predicts)
    return model


def build_self_attention_model_all_level(config, cache):
    embedding_matrix, max_sentences, max_words = cache
    gru_cells = config['gru_cells']
    output_shape = config['output_shape']
    n_words, embedding_dim = embedding_matrix.shape
    embedding_layer = Embedding(input_dim=n_words,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                trainable=False)

    # Input single sentence
    sentence_input = Input((max_words,), dtype='int32')
    embedded_sentence = embedding_layer(sentence_input)
    sentence_att = self_attention_block(embedded_sentence, embedding_dim)
    sentence_lstm = Bidirectional(CuDNNGRU(gru_cells, return_sequences=False))(sentence_att)
    sent_encoder = Model(sentence_input, sentence_lstm)

    # Paragraph level
    review_input = Input(shape=(max_sentences, max_words), dtype='int32')
    review_encoder = TimeDistributed(sent_encoder)(review_input)
    l_lstm_sent = Bidirectional(CuDNNGRU(gru_cells, return_sequences=True))(review_encoder)
    predicts = list()
    for target in targets:
        # att = Dense(1, activation='sigmoid')(l_lstm_sent)
        # weight = Reshape((-1,))(att)
        # weight = RepeatVector(gru_cells*2)(weight)
        # weight = Permute((2, 1))(weight)
        # l_lstm_att = Multiply()([l_lstm_sent, weight])
        # l_lstm_att = BatchNormalization()(l_lstm_att)
        l_lstm_att = self_attention_block(l_lstm_sent, gru_cells * 2)
        l_lstm = Bidirectional(CuDNNGRU(100, return_sequences=False))(l_lstm_att)
        output = Dense(output_shape, activation='softmax', name=target)(l_lstm)
        predicts.append(output)
    model = Model(review_input, predicts)
    return model


def decompose_targets(data):
    decomposed_list = []
    for idx, _ in enumerate(targets):
        target_list = [x[idx] for x in data]
        target_array = np.vstack(target_list)
        decomposed_list.append(target_array)
    return decomposed_list


def main():
    print('----------Loading Data----------')
    train_dataset, validate_dataset, test_dataset = load_dataset()
    train_sentences, train_targets = zip(*train_dataset)
    validate_sentences, validate_targets = zip(*validate_dataset)
    test_sentences, _ = zip(*test_dataset)

    print('----------decompose targets')
    train_targets = decompose_targets(train_targets)
    validate_targets = decompose_targets(validate_targets)

    # Load w2v
    print('----------Loading Embeddings----------')
    w2v = KeyedVectors.load_word2vec_format(data_path_config['embedding_path'],
                                            binary=True, unicode_errors='ignore')
    tok2idx = build_tok2idx(w2v)

    # build embedding matrix
    print('----------Build Embedding Matrix----------')
    w2v_matrix = build_embedding_matrix(tok2idx, w2v, data_path_config['embedding_dim'])

    # convert sentences to array
    print('----------Convert Sentences To Array----------')
    train_array = sentence2array(train_sentences, tok2idx)
    validate_array = sentence2array(validate_sentences, tok2idx)
    test_arrray = sentence2array(test_sentences, tok2idx)

    config = {'gru_cells': 100,
              'output_shape': 4}
    cache = (w2v_matrix, 240, 40)

    print('----------Modeling Time----------')
    model = build_self_attention_model_output_update(config, cache)
    model.compile('sgd', 'categorical_crossentropy', metrics=['acc', ])
    print('------create callbacks')
    model_name = 'HATT_ATTENTION_ALL_OUTPUT_TWO'
    lr_schedule = generate_learning_rate_schedule(0.001, 0.1, 20, 0)
    checkpoint = generate_check_point(model_name)
    early_stopping = generate_early_stopping()
    tensorboard = generate_tensorboard(model_name, 'ALL')
    callbacks = [lr_schedule, tensorboard, early_stopping, checkpoint]
    model.fit(train_array, train_targets, 32, 100,
              verbose=1,
              validation_data=(validate_array, validate_targets),
              callbacks=callbacks)
    test_target_pred = model.predict(test_arrray)
    submission_path = '../submissions/' + model_name + '.npy'
    model.save('../models/' + model_name + '.h5')
    np.save(submission_path, test_target_pred)
    return


if __name__ == '__main__':
    main()
