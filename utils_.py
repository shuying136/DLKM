# -*- coding: utf-8 -*-
import csv
import numpy as np
from nltk.corpus import stopwords
import codecs
stop_words = stopwords.words('english') 
import re
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_small(train_name='data\\TREC.train.all',test_name='data\\TREC.test.all'):
    train_data = []
    train_y = []
    test_data = []
    test_y = []
    with codecs.open(train_name, 'r', encoding="latin-1") as f:
        for line in f:
            train_data.append(clean_str(line).split()[1:])
            train_y.append(int(line.split()[0]))
    with codecs.open(test_name, 'r', encoding="latin-1") as f:
        for line in f:
            test_data.append(clean_str(line).split()[1:])
            test_y.append(int(line.split()[0]))
    return train_data,train_y,test_data,test_y
def load_data():
    test_text = []
    test_text_target = []
    train_text = []
    train_text_target = []
    with open('ag_news_csv\\test.csv','r',encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            test_text.append(row[1]+row[2])
            test_text_target.append(int(row[0])-1)
    with open('ag_news_csv\\train.csv','r',encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            train_text.append(row[1]+row[2])
            train_text_target.append(int(row[0])-1)
    '''
    train_text_no_stop_words = []
    test_text_no_stop_words = []
    for row in train_text:
        temp_text = []
        for col in row.split():
            if col not in stop_words:
                temp_text.append(col)
        train_text_no_stop_words.append(' '.join(temp_text))
    for row in test_text:
        temp_text = []
        for col in row.split():
            if col not in stop_words:
                temp_text.append(col)
        test_text_no_stop_words.append(' '.join(temp_text))
    '''
    text_length = [len(row.split()) for row in train_text]
    for row in test_text:
        text_length.append(len(row.split()))
    print(np.mean(text_length))
    print(np.std(text_length))
    return train_text,train_text_target,test_text,test_text_target