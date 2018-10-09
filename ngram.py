#将三个数据集的数据清洗提取（2，3）gram并且存储到一个文件中，用来训练词向量

import pandas as pd

symbols = ['～', '。', '，' , '』', '『', '"','、', '！', '】', '【', '-', '.', '\n', '？', '）','（','；',' ','：','…','^','⊙','o','_','-',')','(','~','','','']

def Ngram(article):
    print(len(article))
    ngram_list = []
    for sentence in article:
        # 去除标点符号
        sent = ''
        for word in sentence:
            if word not in symbols:
                sent = sent + word
        bi_tokens = []
        th_tokens = []
        for index in range(len(sent)-1):
            bi_tokens.append(sent[index:index+2])

        for index in range(len(sent)-2):
            th_tokens.append(sent[index:index+3])

        ngram_list.append(bi_tokens)
        ngram_list.append(th_tokens)
    return ngram_list


def main():
    train_data = pd.read_csv('datasets/sentiment_analysis_trainingset.csv')
    valid_data = pd.read_csv('datasets/sentiment_analysis_validationset.csv')
    test_data = pd.read_csv('datasets/sentiment_analysis_testa.csv')

    print('train_data len '+ str(len(train_data)))
    print('valid_data len '+ str(len(valid_data)))
    print('test_data len '+ str(len(test_data)))

    X = [train_data['content'], valid_data['content'], test_data['content']]

    F = open('trainData.txt','w')

    for x in X:
        data_list = Ngram(x)
        print(len(data_list))
        for gram in data_list:
            F.write(' '.join(gram))
            F.write('\n')
    F.close()

main()