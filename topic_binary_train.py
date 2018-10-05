from keras.layers import Conv1D, \
    Embedding, Input, MaxPool1D, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import Model
from metrics import f1
from preprocessing import *
from train_config import topic_binary_model_config as config
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import RMSprop


def build_model():
    sent_input = Input((config['max_len'],), dtype='int32')
    embedding_matrix = config['embedding_matrix']
    n_word, embedding_dim = embedding_matrix.shape
    sent_embed = Embedding(input_dim=n_word,
                           output_dim=embedding_dim,
                           weights=[embedding_matrix],
                           trainable=False)(sent_input)
    sent_conv = Conv1D(filters=128, kernel_size=5, padding='same')(sent_embed)
    sent_conv = MaxPool1D(pool_size=5)(sent_conv)
    sent_conv = Conv1D(filters=128, kernel_size=5, padding='same')(sent_conv)
    sent_conv = MaxPool1D(pool_size=5)(sent_conv)
    sent_conv = Conv1D(filters=128, kernel_size=5, padding='same')(sent_conv)
    sent_conv = MaxPool1D(pool_size=5)(sent_conv)
    sent_flat = Flatten()(sent_conv)
    sent_dense = Dense(128, activation='relu')(sent_flat)
    preds = Dense(1, activation='sigmoid')(sent_dense)

    model = Model(sent_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(0.1),
                  metrics=['acc', f1])
    return model


def train(X_train, Y_train, X_val, Y_val):
    model = build_model()
    print(model.summary())
    model.fit(X_train, Y_train,
              validation_data=(X_val, Y_val),
              batch_size=64,
              epochs=100, verbose=2)
    return model


def main():
    # load data and set embedding matrix
    print('--------loading data--------')
    train_tokens_list, valid_tokens_list,\
        test_tokens_list, embedding_matrix,\
        train_df, valid_df, test_df = load_data()
    config['embedding_matrix'] = embedding_matrix

    # pad sequences
    print('--------padding sequence--------')
    train_seq = pad_sequences(train_tokens_list, maxlen=config['max_len'], padding='post', truncating='post')
    print(train_seq.shape)

    for target in train_df.columns:
        if target in {'id', 'content'}:
            continue
        print('-------working on target %s--------' % target)
        train_target = (train_df[target] != -2).astype(int).values
        # train_target = to_categorical(train_target)
        # X_train, X_val, Y_train, Y_val = train_test_split(train_seq, train_target, test_size=0.2)
        train(train_seq, train_target, train_seq, train_target)


if __name__ == '__main__':
    main()

