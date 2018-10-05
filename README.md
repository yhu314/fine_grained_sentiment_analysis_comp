# fine_grained_sentiment_analysis_comp
text cnn 模型修改了参数，输出层变成了20个输出合并成一个list，但是数据处理的label未更改
# Glove
Glove是github的源码，参数已经更更改，只需要改变数据的路径就可以
# ngram.py
读取三个数据文件中的所有数据，清洗后转化成（2，3）gram，同时保存在一个文件中，方便后续训练编码