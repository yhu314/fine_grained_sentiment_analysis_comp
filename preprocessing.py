import pandas as pd
import numpy as np
from collections import defaultdict
import gensim
from train_config import data_path_config as config


def build_sentence_list(all_df):
    sentence_list = [list(x) for x in all_df['content']]
    return sentence_list


def build_count_dict(sentence_list):
    count_dict = defaultdict(lambda: 0)
    for sentence in sentence_list:
        for word in sentence:
            count_dict[word] += 1
    return count_dict


def build_dictionary(count_dict, min_count=10, max_count=3000000):
    word_list = [word for word, count in count_dict.items() if count in range(min_count, max_count)]
    word2idx = {k: v+1 for v, k in enumerate(word_list)}
    word2idx['UNKNOWN'] = len(word_list) + 1
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def reproduce_sentence(sentence_list, word2idx):
    for sentence in sentence_list:
        for idx, word in enumerate(sentence):
            if word not in word2idx:
                sentence[idx] = 'UNKNOWN'
    return sentence_list


def build_prediction_target(train_df, topic):
    targets = (train_df[topic] != -2).astype(int)
    print('%d out of %d comments mentioned topic %s'
          % (np.sum(targets), train_df.shape[0], topic))
    return targets


def build_embedding_matrix(word2idx, keyed_vector, embedding_dim=200):
    n_words = len(word2idx)+1
    embedding_matrix = np.zeros((n_words, embedding_dim), dtype=float)
    for word, idx in word2idx.items():
        if word not in keyed_vector.vocab:
            embedding_matrix[idx, :] = np.random.randn(embedding_dim)
            continue
        embedding_matrix[idx, :] = keyed_vector[word]
    return embedding_matrix


def tokenize(sent_list, word2idx):
    tokens_list = []
    for sentence in sent_list:
        tokens = []
        for word in sentence:
            tokens.append(word2idx.get(word, word2idx['UNKNOWN']))
        tokens_list.append(tokens)
    return tokens_list


def load_data():
    # Load path and other settings
    train_path = config['train_data_path']
    valid_path = config['valid_data_path']
    test_path = config['test_data_path']
    embedding_path = config['embedding_path']
    embedding_dim = config['embedding_dim']

    # Load data
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)
    keyed_vector = gensim\
        .models\
        .KeyedVectors\
        .load_word2vec_format(embedding_path,
                              binary=True,
                              unicode_errors='ignore')

    # Convert String Sentence to list
    train_sent_list = build_sentence_list(train_df)
    valid_sent_list = build_sentence_list(valid_df)
    test_sent_list = build_sentence_list(test_df)

    # Build dictionary
    count_dict = build_count_dict(train_sent_list[:]+valid_sent_list[:]+test_sent_list[:])
    word2idx, idx2word = build_dictionary(count_dict)

    # Build embedding matrix based on loaded word2vec model
    embedding_matrix = build_embedding_matrix(word2idx,
                                              keyed_vector=keyed_vector,
                                              embedding_dim=embedding_dim)

    # Tokenize words based on word2idx dictionary
    train_sent_list_tokenized = tokenize(train_sent_list, word2idx)
    valid_sent_list_tokenized = tokenize(valid_sent_list, word2idx)
    test_sent_list_tokenized = tokenize(test_sent_list, word2idx)
    return train_sent_list_tokenized, valid_sent_list_tokenized,\
           test_sent_list_tokenized, embedding_matrix,\
           train_df, valid_df, test_df


if __name__ == '__main__':
    load_data()
