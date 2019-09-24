import os
import json
import numpy as np
import pandas as pd
import jieba
import tensorflow as tf
import argparse
from dnn import dnn
from gensim import corpora, models

class tf_idf_dnn_predictor:
    def __init__(self, arg):
        self.arg = arg
        with open(self.arg.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.__stop_word_path = self.config['stop_word_path']

        self.load_tf_idf_model()

        self.graph = tf.Graph()
        with self.graph.as_default():
            # 创建模型
            self.model = self.create_model()
            # 加载计算图
            self.sess = self.load_graph()


    def load_tf_idf_model(self):
        self.dictionary = corpora.Dictionary.load("output/tf_idf_model/tf_idf_dict.dict")
        self.tf_idf_model = models.TfidfModel.load("output/tf_idf_model/tf_idf_model.model")
        self.tf_idf_len = len(self.dictionary)


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


    def create_model(self):
        model = dnn(config=self.config, tf_idf_size=self.tf_idf_len, class_nums=2)
        return model


    def load_graph(self):
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.config['ckpt_model_path'])
        print(ckpt)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reloading model parameters...")
            self.model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("No such file :[{}]".format(self.config["ckpt_model_path"]))
        return sess


    def load_test_set(self):
        """
        载入测试用例
        :return:
        """
        sentences = []
        sentences_id = []
        with open(self.config['test_data'], 'r', encoding='utf-8') as f:
            print("<<<< loading test set ..... >>>>>")
            data = f.readlines()
            for line in data[1:]:
                line_split = line.split(',')
                if len(line_split[0]) != 32:
                    print(line_split)
                    raise ValueError(" 这个测试用例有问题 ")

                sentence_need_cut = ''.join(line_split[1:-1])
                sentences_id.append(line_split[0])
                sentence= jieba.lcut(sentence_need_cut)
                sentences.append(sentence)

        return sentences_id, sentences


    def remove_stop_word(self, inputs):

        if self.__stop_word_path:
            stop_words_all = set()
            for path in self.__stop_word_path:
                with open(path, 'r', encoding='utf-8') as f:
                    stop_words = set([line.strip() for line in f.readlines()])
                    stop_words_all = stop_words_all.union(stop_words)
            inputs = [[token for token in sentence if token not in stop_words_all] for sentence in inputs ]

        return inputs

    def predict(self):
        sentences_id, sentences = self.load_test_set()
        sentences_remove_stop_word = self.remove_stop_word(sentences)
        sentences_tf_idf = self.trans_to_tf_idf(input=sentences_remove_stop_word, dictionary=self.dictionary, tf_idf_object=self.tf_idf_model)
        prediction = self.model.infer(self.sess,sentences_tf_idf)
        prediction = list(prediction[0])

        data_frame = pd.DataFrame({'id':sentences_id, 'label':prediction})
        data_frame.to_csv(self.config['test_data_output'], index=False, sep=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', help='model config')
    arg = parser.parse_args()
    predictor = tf_idf_dnn_predictor(arg=arg)
    predictor.predict()
