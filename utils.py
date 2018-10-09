"""
Utility file for processing data
"""
import numpy as np
import os


def build_tok2idx(w2v):
    tok2idx = {}
    idx = 1
    for tok in w2v.vocab:
        if tok == '':
            continue
        tok2idx[tok] = idx
        idx += 1
    return tok2idx


def build_embedding_matrix(tok2idx, w2v, embedding_dim):
    n_words = len(tok2idx)
    embedding_matrix = np.zeros((n_words+1, embedding_dim), dtype=float)
    for tok, idx in tok2idx.items():
        embedding_matrix[idx, :] = w2v[tok]
    return embedding_matrix


def texts_to_sequences(texts, tok2idx):
    sequences = []
    for text in texts:
        tokens = text.split(' ')
        sequence = []
        for token in tokens:
            if token not in tok2idx:
                continue
            sequence.append(tok2idx[token])
        sequences.append(sequence)
    return sequences


def save_model(model, model_name, target):
    model_dir = os.path.join(model_name, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, target+'.h5')
    model.save(model_path)
    return
