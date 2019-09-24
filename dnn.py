import tensorflow as tf

class dnn:
    def __init__(self, config, tf_idf_size, class_nums):
        self.config = config
        self.tf_idf_size = tf_idf_size
        self.class_nums = class_nums

        # placeholder
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.tf_idf_size], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None], name='input_y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        # 超参数
        self.hidden_size_1 = self.config['hidden_size_1']
        self.hidden_size_2 = self.config['hidden_size_2']
        self.hidden_size_3 = self.config['hidden_size_3']
        self.hidden_size_4 = self.config['hidden_size_4']
        self.lr = self.config['learning_rate']
        self.gradient_norm = self.config['gradient_norm']

        # init model
        self.model()
        self.init_saver()


    def model(self):

        with tf.name_scope("DNN"):
            self.hidden_layer_1 = tf.layers.dense(self.input_x, units = self.hidden_size_1, activation = tf.nn.elu)
            self.hidden_layer_1 = tf.nn.dropout(self.hidden_layer_1, keep_prob=self.keep_prob)

            self.hidden_layer_2 = tf.layers.dense(self.hidden_layer_1, units = self.hidden_size_2, activation= tf.nn.elu)
            self.hidden_layer_2 = tf.nn.dropout(self.hidden_layer_2, keep_prob=self.keep_prob)

            self.hidden_layer_3 = tf.layers.dense(self.hidden_layer_2, units = self.hidden_size_3, activation = tf.nn.elu)
            self.hidden_layer_3 = tf.nn.dropout(self.hidden_layer_3, keep_prob=self.keep_prob)

            self.hidden_layer_4 = tf.layers.dense(self.hidden_layer_3, units = self.hidden_size_4, activation= tf.nn.elu)
            self.hidden_layer_4 = tf.nn.dropout(self.hidden_layer_4, keep_prob=self.keep_prob)

            self.hidden_layer_5 = tf.layers.dense(self.hidden_layer_4, units = self.class_nums, activation= tf.nn.elu)

        with tf.name_scope('soft_max'):
            self.softmax = tf.nn.softmax(self.hidden_layer_3, axis=-1)
            self.pred = tf.argmax(self.softmax, axis = -1)
            self.pred = tf.squeeze(self.pred)

        with tf.name_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.hidden_layer_3)
            self.loss = tf.reduce_mean(loss)

        with tf.name_scope('train_op'):
            optimizer = self.get_optimizer()
            train_param = tf.trainable_variables()
            gradients = tf.gradients(self.loss, train_param)
            clips_gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_norm)
            self.train_op = optimizer.apply_gradients(zip(clips_gradients, train_param))


    def get_optimizer(self):
        optimizer = None
        if self.config['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])
        if self.config['optimizer'] == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config['learning_rate'])
        if self.config['optimizer'] == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config['learning_rate'])

        return optimizer


    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)


    def train(self, sess, batch, keep_prob):
        feed_dict = {
            self.input_x:batch['batch_x'],
            self.input_y:batch['batch_y'],
            self.keep_prob: keep_prob
        }
        _, loss, prediction = sess.run([self.train_op, self.loss, self.pred], feed_dict = feed_dict)
        return loss , prediction


    def eval(self, sess, batch):
        feed_dict = {
            self.input_x:batch['batch_x'],
            self.input_y:batch['batch_y'],
            self.keep_prob: 1.0
        }
        loss, prediction = sess.run([self.loss, self.pred], feed_dict = feed_dict)
        return loss, prediction


    def infer(self, sess, input):
        feed_dict = {
            self.input_x:input,
            self.keep_prob:1.0
        }
        prediction = sess.run([self.pred], feed_dict = feed_dict)
        return prediction