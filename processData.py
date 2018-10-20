# 建立词典
def build_voca():
    Dict = []
    F_out = open('datasets/vocabulary_Chinese.txt','w',encoding='utf-8')
    with open('Tencent_AILab_ChineseEmbedding.txt','r',encoding='utf-8') as F:
        line = F.readline()
        while line:
            try:
                line = F.readline()
                line = line.split()
                Dict.append(line[0])
            except:
                pass
    print('vocabulary length = '+str(len(Dict)))
    F_out.write('\n'.join(Dict))
    print('save vocabulary complete! ')
    F_out.close()

# 输入句子，清洗数据
def clear_data(sentence):
    symbols = [' ','⏰','😂','ಥ','≧','∇','≦','～', '。', '，' , '』', '『', '"','、', '！', '】', '【', '-', '.', '\n', '？', '）','（','；',' ','：','…','^','⊙','o','_','-',')','(','~','','','']
    sent = ''
    # 防止出现空行
    for s in sentence.split():
        for word in s:
            if word not in symbols:
                sent = sent + word

    return sent


# 分词
def cut_sentence(train_file, valid_file, test_file):

    import jieba
    import pandas as pd
    jieba.load_userdict('datasets/vocabulary_Chinese.txt')

    train_data = pd.read_csv(train_file)
    valid_data = pd.read_csv(valid_file)
    test_data = pd.read_csv(test_file)

    print('train_data length '+ str(len(train_data)))
    print('valid_data length '+ str(len(valid_data)))
    print('test_data length '+ str(len(test_data)))

    F_train = open('datasets/train_data_jieba.txt','w',encoding='utf-8')
    F_test = open('datasets/test_data_jieba.txt','w',encoding='utf-8')
    F_valid = open('datasets/valid_data_jieba.txt','w',encoding='utf-8')
    
    num = 0
    for sentence in train_data['content']:
        num+=1
        sentence = clear_data(sentence)
        sentence = jieba.lcut(sentence)
        
        F_train.write(str(num)+' '+' '.join(sentence))
        F_train.write('\n')
        
    print(num)
    F_train.close()
    print('train_data cut complete')


    for sentence in valid_data['content']:
        sentence = clear_data(sentence)
        sentence = jieba.lcut(sentence)
        F_valid.write(' '.join(sentence))
        F_valid.write('\n')
    F_valid.close()
    print('valid_data cut complete')
    
    for sentence in test_data['content']:
        sentence = clear_data(sentence)
        sentence = jieba.lcut(sentence)
        F_test.write(' '.join(sentence))
        F_test.write('\n')
    F_test.close()
    print('test_data cut complete')

def main():
    build_voca()
    cut_sentence('datasets/sentiment_analysis_trainingset.csv','datasets/sentiment_analysis_validationset.csv','datasets/sentiment_analysis_testa.csv')
main()