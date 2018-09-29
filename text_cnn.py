from torch_data import UserCommentDataset, calculate_labels, save_predictions
from train_config import data_path_config, targets
from gensim.models import KeyedVectors
from keras import Model
from keras.layers import Conv1D, MaxPooling1D, Dense, Embedding, \
    Input, Flatten, BatchNormalization, AveragePooling1D, Dropout, Concatenate,\
    SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from utils import *
from keras_callbacks import *
import numpy as np
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.regularizers import l1_l2
from metrics import f1
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import f1_score
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))
K.clear_session()


def constrainedCrossEntropy(ytrue, ypred):
  ypred = K.clip(ypred, 0.0001, 0.99999)
  return categorical_crossentropy(ytrue, ypred)


def build_model_v2(embedding_matrixes, maxlen, regularizer, dropout):
    w2v_embedding_matrix, random_embedding_matrix = embedding_matrixes

    random_embedding = Embedding(input_dim=random_embedding_matrix.shape[0],
                                 output_dim=random_embedding_matrix.shape[1],
                                 weights=[random_embedding_matrix],
                                 trainable=True)
    w2v_embedding = Embedding(input_dim=w2v_embedding_matrix.shape[0],
                              output_dim=w2v_embedding_matrix.shape[1],
                              weights=[random_embedding_matrix],
                              trainable=False)
    sent = Input((maxlen,), dtype='int32')
    sent_random = random_embedding(sent)
    sent_w2v = w2v_embedding(sent)
    sent_embedding = Concatenate(axis=-1)([sent_random, sent_w2v])
    sent_embedding = SpatialDropout1D(0.2)(sent_embedding)

    sent_conv1 = Conv1D(8, 5, padding='same', activation='relu', kernel_regularizer=l1_l2(regularizer))(sent_embedding)
    sent_pool1 = MaxPooling1D(5)(sent_conv1)
    sent_conv2 = Conv1D(8, 10, padding='same', activation='relu', kernel_regularizer=l1_l2(regularizer))(sent_embedding)
    sent_pool2 = MaxPooling1D(5)(sent_conv2)
    sent_conv3 = Conv1D(8, 15, padding='same', activation='relu', kernel_regularizer=l1_l2(regularizer))(sent_embedding)
    sent_pool3 = MaxPooling1D(5)(sent_conv3)
    concat = Concatenate()([sent_pool1, sent_pool2, sent_pool3])
    conv = Conv1D(64, 5)(concat)
    pool = MaxPooling1D(5)(conv)
    conv = Conv1D(128, 5)(pool)
    pool = MaxPooling1D(5)(conv)
    flat = Flatten()(pool)
    dense = Dense(128, activation='relu')(flat)
    # dense = Dropout(dropout)(dense)
    logit = Dense(4, activation='softmax')(dense)
    model = Model(sent, logit)
    return model


def main(maxlen, model_name):
    # Load data path and config
    train_path = data_path_config['train_data_path']
    valid_path = data_path_config['valid_data_path']
    test_path = data_path_config['test_data_path']
    submission_path = data_path_config['submission_path']
    embedding_path = data_path_config['embedding_path']
    w2v = KeyedVectors.load_word2vec_format(embedding_path, binary=True, unicode_errors='ignore')

    # Build dict and embedding matrix
    tok2idx = build_tok2idx(w2v)
    embedding_matrix = np.random.randn(len(tok2idx)+1, 200)
    embedding_matrix[0, :] = 0

    w2v_embedding = np.zeros((len(tok2idx)+1, 200))
    for tok, idx in tok2idx.items():
        w2v_embedding[idx, :] = w2v[tok]

    test_data = UserCommentDataset(test_path, None, content='jieba_seg', binary=False)
    for target in targets:
        # clear Keras session
        K.clear_session()

        # Load Data
        train_data = UserCommentDataset(train_path, target, content='jieba_seg', binary=False)
        valid_data = UserCommentDataset(valid_path, target, content='jieba_seg', binary=False)

        # Convert to training and test data
        X_train, Y_train = zip(*train_data)
        X_valid, Y_valid = zip(*valid_data)
        X_test, _ = zip(*test_data)

        # Convert target data
        Y_train = np.asarray(Y_train, dtype=float)
        Y_valid = np.asarray(Y_valid, dtype=float)

        # Convert sentences to sequences
        X_train_seq = texts_to_sequences(X_train, tok2idx)
        X_valid_seq = texts_to_sequences(X_valid, tok2idx)
        X_test_seq = texts_to_sequences(X_test, tok2idx)

        # Pad sequence
        X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
        X_valid_pad = pad_sequences(X_valid_seq, maxlen=maxlen)
        X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

        # generate callbacks for training
        lr_schedule = generate_learning_rate_schedule(0.001, 0.1, 10, 0)
        checkpoint = generate_check_point(model_name+'_'+target)
        early_stopping = generate_early_stopping()
        callbacks = [lr_schedule, checkpoint, early_stopping]

        for regularizer in (0.0001,):
            for dropout in (0.3,):
                print('--------Working on target {}--------'.format(target))
                print('-----regularizer {}, dropout {}----'.format(regularizer, dropout))
                model = build_model_v2((w2v_embedding, embedding_matrix), maxlen, regularizer, dropout)
                model.compile(optimizer='sgd',
                              loss=constrainedCrossEntropy, metrics=['acc', f1])
                model.fit(X_train_pad, Y_train, validation_data=(X_valid_pad, Y_valid),
                          callbacks=callbacks, verbose=2,
                          epochs=100, batch_size=64)

                Y_valid_pred_prob = model.predict(X_valid_pad, batch_size=64)
                Y_test_pred_prob = model.predict(X_test_pad, batch_size=64)
                Y_valid_pred = calculate_labels(Y_valid_pred_prob)
                print('----F1 Score for validation set----')
                print(f1_score(Y_valid, Y_valid_pred, average='macro'))
                save_predictions(Y_test_pred_prob, target,submission_path)
    return


if __name__ == '__main__':
    maxlen = 2890
    model_name = 'text_cnn_w2v/simple_textcnn_category_sgd_spatial'
    main(maxlen,model_name)