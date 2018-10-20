import numpy as np
import pickle


output_embedding_path = '../tencent_embedding/embedding_array.npy'
output_vocab_path = '../tencent_embedding/vocab.pickle'


def main():
    path = '../tencent_embedding/Tencent_AILab_ChineseEmbedding.txt'
    vocab = dict()
    embeddings = np.empty((0, 200), dtype=float)
    with open(path, 'rb') as file:
        while True:
            line = file.readline()
            if not line:
                break

            try:
                line = line.decode('utf-8')
            except:
                print(line)
                continue

            line_split = line.split(' ')
            word, embed_string = line_split[0], line_split[1:]
            vocab[word] = len(vocab)
            embed_array = np.array(embed_string, dtype=float)
            if len(embed_array) != 200:
                print(word)
                continue

            embeddings = np.vstack((embeddings, embed_array))
            if len(vocab) % 10000 == 0:
                print('{} lines parsed'.format(len(vocab)))

    np.save(output_embedding_path, embeddings)
    with open(output_vocab_path, 'r') as file:
        pickle.dump(vocab, file)
    return


if __name__ == '__main__':
    main()