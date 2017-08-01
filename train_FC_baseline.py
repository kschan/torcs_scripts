import tensorflow as tf
import numpy as np
import glob

def concatenate_training_data():
    logged_input_files = glob.glob("good_logs/logged_inputs*")
    logged_inputs = np.concatenate([np.load(file) for file in logged_input_files], axis = 0)
    print("logged_inputs: ", logged_inputs.shape)

    logged_state_files = glob.glob("good_logs/logged_states*")
    logged_states = np.concatenate([np.load(file) for file in logged_state_files], axis = 0)
    print("logged_states: ", logged_states.shape)

    return logged_states, logged_inputs



logged_states, logged_inputs = concatenate_training_data()
num_samples = logged_states.shape[0]
random_indices = np.random.permutation(num_samples)
train_indices = random_indices[:40000]
valid_indices = random_indices[40001:50000]
test_indices = random_indices[50001:]

train_dataset = logged_states[train_indices, :]
train_labels = logged_inputs[train_indices, 0].reshape((-1, 1))
print("train_labels: ", train_labels.shape)

valid_dataset = logged_states[valid_indices, :]
valid_labels = logged_inputs[valid_indices, 0].reshape((-1, 1))

test_dataset = logged_states[test_indices, :]
test_labels = logged_inputs[test_indices, 0].reshape((-1, 1))

num_states = train_dataset.shape[1]
num_inputs = 1

# NN

batch_size = 16
beta = 0.
graph = tf.Graph()

layer_sizes = [256]*4

input_dataset = tf.placeholder(tf.float32, shape = (None, num_states))
input_labels = tf.placeholder(tf.float32, shape = (None, num_inputs))
bn_input = tf.layers.batch_normalization(input_dataset)
# Variables
layer1 = tf.layers.dense(bn_input, units = layer_sizes[0], activation = tf.nn.relu,
    kernel_initializer = tf.truncated_normal_initializer)

layer2 = tf.layers.dense(inputs = layer1, units = layer_sizes[1], activation = tf.nn.relu,
    kernel_initializer = tf.truncated_normal_initializer)

layer3 = tf.layers.dense(inputs = layer2, units = layer_sizes[2], activation = tf.nn.relu,
    kernel_initializer = tf.truncated_normal_initializer)

layer4 = tf.layers.dense(inputs = layer3, units = layer_sizes[3], activation = tf.nn.relu,
    kernel_initializer = tf.truncated_normal_initializer)

# layer5 = tf.layers.dense(layer4, units = layer5_size, activation = tf.nn.relu,
#     kernel_initializer = tf.truncated_normal_initializer)

output = tf.layers.dense(inputs = layer4, units = 1, activation = tf.nn.tanh,
    kernel_initializer = tf.truncated_normal_initializer)

W_matrix_layer4 = [v for v in tf.global_variables() if v.name == "dense_4/kernel:0"][0]

loss = tf.losses.mean_squared_error(labels=input_labels, predictions=output)
  
optimizer = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss)

num_steps = 30000

with tf.Session() as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for var in tf.global_variables():
        print(var.name)
    
    for step in range(num_steps):
        batch_indices = np.random.randint(train_labels.shape[0], size = batch_size)
        batch_dataset = train_dataset[batch_indices, :]
        batch_labels = train_labels[batch_indices, :]
        feed_dict = {input_dataset:batch_dataset, input_labels:batch_labels}
        _, l, p = session.run([optimizer, loss, output], feed_dict = feed_dict)

        
        if (step%500 == 0):
            feed_dict = {input_dataset:valid_dataset, input_labels:valid_labels}
            o, vl, wv4 = session.run([output, loss, W_matrix_layer4], feed_dict = feed_dict)
            print("(step, loss, val_loss): (%d, %f, %f)" % (step, l, vl))
            print(sum(sum(wv4)))
            
    feed_dict = {input_dataset:test_dataset, input_labels:test_labels}
    l = session.run(loss, feed_dict = feed_dict)
    print("test loss: %f" % l)