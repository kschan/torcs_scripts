import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, model_name): 
       
        if model_name == 'fc_steer':
            self.build_fc_steer()

    def build_fc_steer(self):

        self.input_idxs = [3]
        self.states_idxs = [0, 54, 63, 72, 73]

        self.num_inputs = len(self.input_idxs)
        self.num_states = len(self.states_idxs)

        # Create the model

        self.x = tf.placeholder(tf.float32, [None, self.num_states], name="x")
        self.y = tf.placeholder(tf.float32, [None, self.num_inputs], name="y")

        x_norm = tf.nn.l2_normalize(self.x, dim = 1, epsilon=1e-12, name = 'x_norm')
        # y_norm = tf.nn.l2_normalize(self.y, dim = 1, epsilon=1e-12, name = 'y_norm')

        fc1 = tf.nn.tanh(tf.layers.dense(x_norm, 10))
        # fc1 = tf.layers.batch_normalization(fc1)

        # fc2 = tf.nn.tanh(tf.layers.dense(fc1, 256))
        # fc2 = tf.layers.batch_normalization(fc2)

        self.predictions = tf.layers.dense(fc1, 1, name="predictions")

        with tf.name_scope('loss'):
            self.mse_loss = tf.losses.mean_squared_error(labels = self.y, predictions = self.predictions)

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.mse_loss)

        