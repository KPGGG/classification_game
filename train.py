import tensorflow as tf
import json
import os
import argparse
import random
from data_helper import tf_idf_data_helper
from dnn import dnn
from metric import metric

metric = metric()
rate = 0.8

class DNN_trainer:
    def __init__(self, arg):
        self.arg = arg
        with open(self.arg.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # data_helper 定义
        self.data_loader_obj = tf_idf_data_helper(stop_word_path= self.config['stop_word_path'], low_freq=self.config['low_freq'])

        # all data
        self.all_data, self.all_labels = self.data_loader_obj.gen_data(self.config['train_data'])
        self.tf_idf_len = self.data_loader_obj.dictionary_len
        self.class_nums = self.data_loader_obj.class_nums
        print("<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")
        print("class nums :{}".format(self.class_nums))
        print("<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>")

        # train data
        self.train_data_len = int(len(self.all_data)*rate)
        self.train_data, self.train_labels = self.all_data[:self.train_data_len], self.all_labels[:self.train_data_len]

        # eval data
        self.eval_data, self.eval_labels = self.all_data[self.train_data_len:], self.all_labels[self.train_data_len:]

        # 模型定义
        self.model_obj = dnn(config=self.config, tf_idf_size=self.tf_idf_len, class_nums=self.class_nums)


    def train(self):
        gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9, allow_growth = True)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement= True, gpu_options = gpu_option)
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            current_step = 0

            for epoch in range(self.config['epoch']):
                print(" -------------epoch : {} / {}----------------".format(epoch+1, self.config['epoch']))

                for batch in self.data_loader_obj.next_batch(x=self.train_data,
                                                             y=self.train_labels,
                                                             batch_size=self.config['batch_size']):
                    loss, prediction = self.model_obj.train(sess, batch=batch, keep_prob=self.config['keep_prob'])
                    acc = metric.accuracy(batch['batch_y'], prediction)
                    f1 = metric.f1(batch['batch_y'], prediction)
                    precision = metric.precision(batch['batch_y'], prediction)
                    recall = metric.recall(batch['batch_y'], prediction)
                    print('current_step:{}, loss:{}, acc:{}, f1:{}, precision:{}, recall:{}'.format(current_step, loss, acc, f1, precision, recall))
                    current_step += 1

                    if current_step % self.config['check_point'] == 0:
                        batch_eval = {'batch_x':self.eval_data, 'batch_y':self.eval_labels}
                        eval_loss, eval_pred = self.model_obj.eval(sess,batch_eval)
                        eval_acc = metric.accuracy(batch_eval['batch_y'], eval_pred)
                        eval_f1 = metric.f1(batch_eval['batch_y'], eval_pred)
                        eval_precision = metric.precision(batch_eval['batch_y'], eval_pred)
                        eval_recall = metric.recall(batch_eval['batch_y'], eval_pred)
                        print("<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>")
                        print("<<<< eval_acc:{}, eval_f1:{}, eval_precision:{}, eval_recall:{}".format(eval_acc, eval_f1, eval_precision, eval_recall))
                        print("<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>")
                        if self.config['ckpt_model_path']:
                            save_path = os.path.join(os.path.join(os.getcwd()),
                                                     self.config['ckpt_model_path'])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config['model_name'])
                            self.model_obj.saver.save(sess, model_save_path, global_step=current_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', help='model config')
    args = parser.parse_args()
    trainer = DNN_trainer(args)
    trainer.train()