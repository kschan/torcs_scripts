import numpy as np
import tensorflow as tf

class Model:
    def __init__(self, model_name, INPUT, OUTPUT): 
        if model_name == 'fc':
            print('[ INFO ] Building fully connected network ...')
            self.build_fc(INPUT, OUTPUT)
        else:
            raise RuntimeError("[ ERROR ] Wrong model_name entered")

    def build_fc(self, INPUT, OUTPUT):
        # Create placeholder
        self.x = tf.placeholder(tf.float32, [None, len(INPUT)], name="x")
        self.y = tf.placeholder(tf.float32, [None, len(OUTPUT)], name="y")

        # Network architecture
        fc1 = tf.nn.relu(tf.layers.dense(self.x, 32))
        fc2 = tf.nn.relu(tf.layers.dense(fc1, 32))
        self.predictions = tf.layers.dense(fc2, len(OUTPUT), name="predictions")

        # Define loss and optimizer
        with tf.name_scope('loss'):
            self.total_loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.predictions)

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-8).minimize(self.total_loss)
