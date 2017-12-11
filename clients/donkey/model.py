import numpy as np
import tensorflow as tf

class Model:
    def __init__(self, model_name, INPUT=None, OUTPUT=None): 
        if model_name == 'fc':
            print('[ INFO ] Building fully connected network ...')
            self.build_fc(INPUT, OUTPUT)
        elif model_name == 'steer':
            print('[ INFO ] Building steer network ...')
            self.build_steer_net()
        elif model_name == 'accel':
            print('[ INFO ] Building accel network ...')
            self.build_accel_net()
        elif model_name == 'steer_accel':
            print('[ INFO ] Building steer_accel network ...')
            self.build_steer_accel_net()
        elif model_name == 'separate':
            print('[ INFO ] Building separate network ...')
            self.build_sepearate_net(INPUT, OUTPUT)
        else:
            raise RuntimeError("[ ERROR ] Wrong model_name entered")

    def build_fc(self, INPUT, OUTPUT):
        # Create placeholder
        self.x = tf.placeholder(tf.float32, [None, len(INPUT)], name="x")
        self.y = tf.placeholder(tf.float32, [None, len(OUTPUT)], name="y")

        # Network architecture
        fc1 = tf.nn.tanh(tf.layers.dense(self.x, 5, name='fc1'))

        self.predictions = tf.layers.dense(fc1, len(OUTPUT), name="predictions")

        # Define loss and optimizer
        with tf.name_scope('loss'):
            self.total_loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.predictions)

        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0, trainable=False, name='step')
            self.train_step = tf.train.AdamOptimizer(1e-7).minimize(self.total_loss, global_step=self.global_step)

    def build_steer_net(self):
        # Create placeholder
        self.x = tf.placeholder(tf.float32, [None, 2], name="x") # angle and trackPos
        self.y = tf.placeholder(tf.float32, [None, 1], name="y")
    
        # Network architecture
        fc1 = tf.nn.tanh(tf.layers.dense(self.x, 5))
        self.predictions = tf.nn.tanh(tf.layers.dense(fc1, 1, name="predictions"))                   
    
        # Define loss and optimizer
        with tf.name_scope('loss'):
            self.total_loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.predictions)
    
        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-5).minimize(self.total_loss)

    def build_accel_net(self):
        # Create placeholder
        self.x = tf.placeholder(tf.float32, [None, 4], name="x_accel") # angle, trackPos, speedX, steer 
        self.y = tf.placeholder(tf.float32, [None, 1], name="y_accel")
    
        # Network architecture for steer
        fc1_steer = tf.nn.tanh(tf.layers.dense(self.x, 5, name="fc1_accel"))
        self.predictions = tf.nn.tanh(tf.layers.dense(fc1, 1, name="pred_accel"))                   
    
        # Define loss and optimizer
        with tf.name_scope('loss_accel'):
            self.loss_accel = tf.losses.mean_squared_error(labels=self.y, predictions=self.predictions)
    
        with tf.name_scope('opt_accel'):
            self.opt_accel = tf.train.AdamOptimizer(1e-5).minimize(self.loss_accel)

    def build_steer_accel_net(self):
        # Create placeholder
        self.x_steer = tf.placeholder(tf.float32, [None, 2], name="x_steer") # angle, trackPos
        self.y_steer = tf.placeholder(tf.float32, [None, 1], name="y_steer")

        self.x_accel = tf.placeholder(tf.float32, [None, 3], name="x_accel") # angle, trackPos, speedX
        self.y_accel = tf.placeholder(tf.float32, [None, 1], name="y_accel")
    
        # Network architecture
        fc1_steer       = tf.nn.tanh(tf.layers.dense(self.x_steer, 5, name="fc1_steer"))
        self.pred_steer = tf.nn.tanh(tf.layers.dense(fc1_steer, 1, name="pred_steer"))                   

        accel_input     = tf.concat([self.x_accel, self.pred_steer], axis=1) # angle, trackPos, speedX, steer 
        fc1_accel       = tf.nn.tanh(tf.layers.dense(accel_input, 5, name="fc1_accel"))
        self.pred_accel = tf.layers.dense(fc1_accel, 1, name="pred_accel")
    
        # Define loss and optimizer
        with tf.name_scope('loss'):
            self.loss_steer = tf.losses.mean_squared_error(labels=self.y_steer, predictions=self.pred_steer)
            self.loss_accel = tf.losses.mean_squared_error(labels=self.y_accel, predictions=self.pred_accel)
            self.total_loss = self.loss_steer + 0.2*self.loss_accel
    
        with tf.name_scope('opt'):
            self.opt = tf.train.AdamOptimizer(1e-5).minimize(self.total_loss)

    def build_sepearate_net(self, INPUT, OUTPUT):
        # Create placeholder
        self.x = tf.placeholder(tf.float32, [None, len(INPUT)], name="x_steer")

        self.y_steer = tf.placeholder(tf.float32, [None, 1], name="y_steer")
        self.y_accel = tf.placeholder(tf.float32, [None, 1], name="y_accel")

        # Network architecture
        fc1_steer = tf.nn.tanh(tf.layers.dense(self.x, 5))
        self.pred_steer = tf.layers.dense(fc1_steer, 1)

        fc1_accel = tf.nn.tanh(tf.layers.dense(self.x, 5))
        fc2_accel = tf.nn.tanh(tf.layers.dense(fc1_accel, 5))
        self.pred_accel = tf.layers.dense(fc2_accel, 1)

        self.predictions = tf.concat([self.pred_steer, self.pred_accel], axis=1)

        # Define loss and optimizer
        with tf.name_scope('loss'):
            self.loss_steer = tf.losses.mean_squared_error(labels=self.y_steer, predictions=self.pred_steer)
            self.loss_accel = tf.losses.mean_squared_error(labels=self.y_accel, predictions=self.pred_accel)
            # self.total_loss = self.loss_steer + self.loss_accel

        with tf.name_scope('optimizer'):
            self.global_step_steer = tf.Variable(0, trainable=False, name='step')
            self.opt_steer = tf.train.AdamOptimizer(1e-7).minimize(self.loss_steer, global_step=self.global_step_steer)

            self.global_step_accel = tf.Variable(0, trainable=False, name='step')
            self.opt_accel = tf.train.AdamOptimizer(1e-7).minimize(self.loss_accel, global_step=self.global_step_accel)
