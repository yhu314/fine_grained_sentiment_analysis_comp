import re


def remove_special_tokens(sent):
    tokens = {'*', '^', 'O'}
    for token in tokens:
        sent = sent.replace(token, '')
    return sent


def process_error_seg(sent):
    repls = [('(是\s吗\s*){1,}', '是吗'),
             ('(去\s吧\s*){1,}', '去吧'),
             ('(坑\s爹\s*){1,}', '坑爹')]
    for pattern, repl in repls:
        sent = re.sub(pattern, repl, sent)
    return sent


def split_sentence(sent):
    splits = ';+|。+|！+|…+|\r+|\n+|，+|~+|!+|\s{2,}|,+|\.{1,}|～+|、+|\'+|≡+|·+|-+|〜'
    return re.split(splits, sent)


def process_sentence(sents):
    processed_sents = []
    void_set = {' ', '', '*', '^', 'O'}
    for sent in sents:
        sent = sent.lstrip(' ').rstrip(' ')
        if sent in void_set:
            continue
        processed_sents.append(sent)
    return processed_sents


def remove_dups(sents):
    processed_sents = []
    appeared_sents = set()
    for sent in sents:
        if sent in appeared_sents:
            continue
        appeared_sents.add(sent)
        processed_sents.append(sent)
    return processed_sents