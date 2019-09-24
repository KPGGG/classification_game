import jieba
import os
import random
import numpy as np
from collections import Counter
from itertools import chain
from gensim import corpora, models

class tf_idf_data_helper:
    def __init__(self,
                 stop_word_path = None,
                 tf_idf_model_save_path=None,
                 low_freq = 1,
                 train=True ):

        self.train = train
        self.__stop_word_path = stop_word_path
        self.__tf_idf_model_save_path = tf_idf_model_save_path
        self.__low_freq = low_freq
        self.dictionary = None
        self.tf_idf_model = None
        self.count = 0


    def load_data(self, file_path):
        sentences = []
        labels = []
        print("loading data...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in data[1:]:
                line_split = line.split(",")
                #if len(line_split)<= 2:
                sentence_need_cut = ''.join(line_split[1:-1])
                sentence = jieba.lcut(sentence_need_cut)
                label = int(line_split[-1])
                sentences.append(sentence)
                labels.append(label)

        return sentences, labels


    def remove_stop_word(self, inputs):
        all_words= list(chain(*inputs))
        print(len(all_words))
        word_count = dict(Counter(all_words))
        inputs = [[ word for word in sentence if word_count[word] > self.__low_freq] for sentence in inputs]

        if self.__stop_word_path:
            stop_words_all = set()
            for path in self.__stop_word_path:
                with open(path, 'r', encoding='utf-8') as f:
                    stop_words = set([line.strip() for line in f.readlines()])
                    stop_words_all = stop_words_all.union(stop_words)
            inputs = [[token for token in sentence if token not in stop_words_all] for sentence in inputs ]

        return inputs

    def train_tf_idf(self, input):
        sentences = input
        dictionary = corpora.Dictionary(sentences)
        corpus = [dictionary.doc2bow(sentence) for sentence in sentences]
        tfidf_model = models.TfidfModel(corpus)
        save_path = "output/tf_idf_model"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        dictionary.save("output/tf_idf_model/tf_idf_dict.dict")
        tfidf_model.save("output/tf_idf_model/tf_idf_model.model")

        return dictionary, tfidf_model


    def trans_to_tf_idf(self, input, dictionary, tf_idf_object):
        vocab_size = len(dictionary.token2id)
        input_ids = []

        for sentence in input:
            bow_vec = dictionary.doc2bow(sentence)
            tfidf_vec = tf_idf_object[bow_vec]
            vec = [0]*vocab_size
            for item in tfidf_vec:
                vec[item[0]] = item[1]
            input_ids.append(vec)

        return input_ids


    def gen_data(self, file_path):
        # 1. 读取数据
        inputs, labels = self.load_data(file_path)

        # 2. 去除停用词
        inputs_remove_stop_word = self.remove_stop_word(inputs)

        # 3. if train 训练tf——idf模型
        self.dictionary, self.tf_idf_model = self.train_tf_idf(inputs_remove_stop_word)
        self.dictionary_len = len(self.dictionary)
        self.class_nums = len(set(labels))

        # 4.输入转tf-idf
        input_ids = self.trans_to_tf_idf(inputs_remove_stop_word, self.dictionary, self.tf_idf_model)

        # 5. labels to index
        labels_to_idx = labels

        return input_ids, labels_to_idx


    def next_batch(self, x, y, batch_size):
        """

        :param x:
        :param y:
        :param batch_size:
        :return:
        """
        num_batches = len(x)//batch_size

        z = list(zip(x,y))
        random.shuffle(z)
        x, y = zip(*z)

        for i in range(num_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = x[start:end]
            batch_y = y[start:end]
            yield dict(batch_x = batch_x, batch_y = batch_y)

