import gensim
import pandas as pd


def main_char_word2vec(contents):
    # process string to list
    sent_list = [' '.join(list(x)) for x in contents]
    emdeding_dim = 200
    model = gensim.models.Word2Vec(
        sent_list,
        size=emdeding_dim,
        window=5,
        min_count=1,
        hs=0,
        iter=10,
        negative=5,
        workers=8
    )
    model_dir = 'word2vec/model_char_%s.kv' % emdeding_dim
    model.wv.save_word2vec_format(model_dir, binary=True)
    return


if __name__ == '__main__':
    train_path = 'sentiment_analysis_trainingset.csv'
    valid_path = 'sentiment_analysis_validationset.csv'
    test_path = 'sentiment_analysis_testa.csv'
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)
    all_df = pd.concat([train_df, valid_df, test_df])
    all_contents = all_df['content'].values
    main_char_word2vec(all_contents)
