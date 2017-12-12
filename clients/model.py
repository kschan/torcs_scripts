import tensorflow as tf
import numpy as np

'''
STATES:
angle           0
curLapTime      1
damage          2
distFromStart   3
distRaced       4
focus           5
fuel            10
gear            11
lastLapTime     12
opponents       13
racePos         49
rpm             50
speedX          51
speedY          52
speedZ          53
track           54 - 72, middle is 63
trackPos        73
wheelSpinVel    74
z               78

INPUTS:
accel           0
brake           1
gear            2
steer           3
clutch          4
'''

class Model:
    def __init__(self, model_name): 
        if model_name == 'fc_steer':
            print('[ INFO ] Building fc_steer ...')
            self.build_fc_steer()
        elif model_name == 'cnn_steer':
            print('[ INFO ] Building cnn_steer ...')
            self.build_cnn_steer()
        elif model_name == 'donkey_steer':
            print('[ INFO ] Building donkey_steer ...')
            self.build_donkey_steer()
        elif model_name == 'steer_accel':
            print('[ INFO ] Building steer_accel ...')
            self.build_steer_accel()
        else:
            raise RuntimeError("[ ERROR ] Wrong model_name entered")

    def build_steer_accel(self):     
        self.input_idxs = [3]       # only steering
        # self.states_idxs = [0, 73] + list(np.arange(54, 73)) + [51]
        self.states_idxs = list(np.arange(54, 73))

        self.num_states  = len(self.states_idxs)
        self.num_inputs = len(self.input_idxs)

        # Create the model
        self.x = tf.placeholder(tf.float32, [None, self.num_states], name="x")
        self.y = tf.placeholder(tf.float32, [None, self.num_inputs], name="y")

        x_steer = self.x/([200.]*19)


        steer_fc1 = tf.nn.relu(tf.layers.dense(x_steer   , 1000, kernel_initializer=tf.initializers.truncated_normal()))
        steer_fc2 = tf.nn.relu(tf.layers.dense(steer_fc1 , 2000, kernel_initializer=tf.initializers.truncated_normal()))
        steer_fc3 = tf.nn.relu(tf.layers.dense(steer_fc2 , 5000, kernel_initializer=tf.initializers.truncated_normal()))

        self.predictions_steer = tf.nn.tanh(tf.layers.dense(steer_fc2 , 1, kernel_initializer=tf.initializers.truncated_normal()))

        # x_accel = tf.concat([self.predictions_steer, speedX], axis = 1)
        # accel_fc1 = tf.nn.relu(tf.layers.dense(x_accel , 5, kernel_initializer=tf.initializers.truncated_normal()))
        # self.predictions_accel = tf.layers.dense(accel_fc1 , 1, kernel_initializer=tf.initializers.truncated_normal())
        self.predictions_accel = tf.constant(0.2)

        # Define loss and optimizer

        with tf.name_scope('loss'):
            self.steer_loss = tf.losses.mean_squared_error(labels=self.y[:, 0], predictions = tf.reshape(self.predictions_steer, [-1]))
            self.total_loss = self.steer_loss

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.total_loss)


    def build_fc_steer(self):
        self.states_idxs = [0,73]
        self.num_states  = len(self.states_idxs)

        self.input_idxs = [3]       # only steering
        self.states_idxs = [54, 63, 72, 73]

        self.input_idxs = [3,0]
        self.num_inputs = len(self.input_idxs)

        # Create the model
        self.x = tf.placeholder(tf.float32, [None, self.num_states], name="x")
        self.y = tf.placeholder(tf.float32, [None, self.num_inputs], name="y")

        # x_norm = tf.nn.l2_normalize(self.x, dim = 1, epsilon=1e-12, name = 'x_norm')
        # y_norm = tf.nn.l2_normalize(self.y, dim = 1, epsilon=1e-12, name = 'y_norm')

        fc1 = tf.nn.tanh(tf.layers.dense(self.x , 5, kernel_initializer=tf.initializers.truncated_normal()))

        steer_fc = tf.nn.tanh(tf.layers.dense(fc1 , 5, kernel_initializer=tf.initializers.truncated_normal()))
        accel_fc = tf.nn.tanh(tf.layers.dense(fc1 , 5, kernel_initializer=tf.initializers.truncated_normal()))
        
        # fc1 = tf.layers.batch_normalization(fc1)

        # fc2 = tf.nn.tanh(tf.layers.dense(fc1, 256))
        # fc2 = tf.layers.batch_normalization(fc2)

        self.predictions_steer  = tf.layers.dense(steer_fc, 1, name="predictions_steer", kernel_initializer=tf.initializers.truncated_normal())
        self.predictions_accel = tf.layers.dense(accel_fc, 1, name="predictions_accel", kernel_initializer=tf.initializers.truncated_normal())

        # Define loss and optimizer
        self.predictions = tf.concat([self.predictions_steer, self.predictions_accel], axis=1)

        with tf.name_scope('loss'):
            self.total_loss = tf.losses.mean_squared_error(labels = self.y, predictions = self.predictions)

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.total_loss)

    def build_cnn_steer(self):
        self.input_idxs = [3]   # only steering
        self.states_idxs = list(np.arange(54, 73))# + [73] + [0]

        self.num_inputs = len(self.input_idxs)
        self.num_states = len(self.states_idxs)

        self.x = tf.placeholder(tf.float32, [None, self.num_states], name="x")
        self.y = tf.placeholder(tf.float32, [None, self.num_inputs], name="y")

        x_norm = (self.x[:,:19]/200.0)   # normalizing track distance readings
        x_norm = tf.reshape(x_norm, [-1, 1, 19, 1])

        conv1 = tf.layers.conv2d(x_norm, filters = 50, kernel_size = [1, 5], strides = [1, 2])
        conv1 = tf.nn.tanh(conv1)

        conv2 = tf.layers.conv2d(conv1, filters = 100, kernel_size = [1, 3], strides=[1, 2])
        conv2 = tf.nn.tanh(conv2)

        flattened = tf.contrib.layers.flatten(conv2)

        fc1_conv = tf.nn.tanh(tf.layers.dense(flattened, 200))

        # fc2_conv = tf.nn.tanh(tf.layers.dense(fc1_conv, 5))

        # track_pos = tf.reshape(self.x[:,19], [-1, 1])
        # angle = tf.reshape(self.x[:, 20], [-1, 1])

        scan_indices = [7, 9, 11]
        scans = tf.gather(self.x, scan_indices, axis=1)
        scans = tf.reshape(scans, [-1, len(scan_indices)])/200.0

        concat = tf.concat([fc1_conv, scans], axis=1)
        fc1 = tf.nn.tanh(tf.layers.dense(fc1_conv, 100))
        self.predictions_steer = tf.nn.tanh(tf.layers.dense(fc1, self.num_inputs))
        self.predictions_accel = tf.constant(0.2)

        with tf.name_scope('loss'):
            self.total_loss = tf.losses.absolute_difference(labels = self.y, predictions = self.predictions_steer)

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1E-6).minimize(self.total_loss)

    def build_donkey_steer(self):
        # Input data
        self.states_idxs = [0, 73, 51]
        self.num_states  = len(self.states_idxs)

        # Output data
        self.output_idxs = [3, 0]
        self.output_idxs = [0]
        self.num_outputs = len(self.output_idxs)

        # Create the model
        self.x = tf.placeholder(tf.float32, [None, self.num_states], name="x")
        self.y = tf.placeholder(tf.float32, [None, self.num_outputs], name="y")

        fc1 = tf.nn.tanh(tf.layers.dense(self.x, 5))

        self.predictions = tf.nn.tanh(tf.layers.dense(fc1, self.num_outputs, name="predictions"))

        # Define loss and optimizer
        with tf.name_scope('loss'):
            self.total_loss = tf.losses.mean_squared_error(labels = self.y, predictions = self.predictions)

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-5).minimize(self.total_loss)

    def build_gan_steer():
        self.input_idxs = [3]   # only steering
        self.states_idxs = list(np.arange(54, 73)) + [73]# + [0]

        self.num_inputs = len(self.input_idxs)
        self.num_states = len(self.states_idxs)

        self.x = tf.placeholder(tf.float32, [None, self.num_states], name="x")
        self.y = tf.placeholder(tf.float32, [None, self.num_inputs], name="y")