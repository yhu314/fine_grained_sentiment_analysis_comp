from transformers import *
from train_config import data_path_config, targets
from torch_data import UserCommentDataset
from gensim.models import KeyedVectors
from utils import build_tok2idx, build_embedding_matrix


def load_dataset():
    transformers = [remove_special_tokens,
                    process_error_seg,
                    split_sentence,
                    process_sentence,
                    remove_dups]
    train_dataset = UserCommentDataset(data_path_config['train_data_path'],
                                       targets, content='jieba_seg', transformers=transformers)
    test_dataset = UserCommentDataset(data_path_config['test_data_path'],
                                       targets, content='jieba_seg', transformers=transformers)
    return train_dataset, test_dataset


def main():
    train_dataset, test_datset = load_dataset()
    train_sent, _, train_targets = zip(*train_dataset)
    test_sent, _, _ = zip(*test_datset)

    # Load w2v
    w2v = KeyedVectors.load_word2vec_format(data_path_config['embedding_path'],
                                            binary=True, unicode_errors='ignore')
    tok2idx = build_tok2idx(w2v)

    # build embedding matrix
    embedding_matrix = build_embedding_matrix(tok2idx, w2v, data_path_config['embedding_dim'])


    return

if __name__ == '__main__':
    main()