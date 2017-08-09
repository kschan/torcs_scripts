"""Various helpers functions for training and evaluation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict
import os
import string
import numpy as np
import tensorflow as tf

def get_input_file_paths(datasets_dir, dataset):
  shards = sorted(os.listdir(datasets_dir))
  return [os.path.join(datasets_dir, shard) for shard in shards]


def check_if_should_stop(model_dir):
  log_path = os.path.join(model_dir, 'train_results.txt')
  if not gfile.Exists(log_path):
    return False
  with gfile.Open(log_path, 'r') as f:
    should_break = False
    for line in f:
      if 'done' in line.strip():
        should_break = True
    return should_break


def copy_ckpt_with_new_var_names(load_path, tables, model, batch_size,
                                 num_epochs, n_symbols, num_layers, state_size,
                                 dropout_keep, prefix, save_path):

  tf.reset_default_graph()
  with tf.Graph().as_default():
    input_x, _, input_seqlens = get_input_producers(tables, batch_size,
                                                    num_epochs)
    # set up old graph
    _ = get_graph(input_x, input_seqlens, model, n_symbols, num_layers,
                  state_size, dropout_keep)
    old_vars = tf.global_variables()

    with tf.variable_scope(prefix):
      # set up graph with prefix
      _ = get_graph(input_x, input_seqlens, model, n_symbols, num_layers,
                    state_size, dropout_keep)
    new_vars = get_vars_by_starting_pattern(prefix, tf.global_variables())

    assign_ops = []
    for var in old_vars:
      new_var = get_vars_by_matching_pattern(var.name, new_vars)[0]
      assign_op = tf.assign(new_var, var)
      assign_ops.append(assign_op)

    old_var_saver = tf.train.Saver(old_vars)
    new_var_saver = tf.train.Saver(new_vars)
    print('new vars for', prefix)
    print_var_collection(new_vars)

    with tf.Session() as sess:
      old_var_saver.restore(sess, load_path)
      sess.run(assign_ops)
      new_var_saver.save(sess, save_path)

    return save_path

def mkdirp(path):
  if not os.path.isdir(path):
    os.makedirs(save_dir)

def get_load_path(base_load_dir, model_config_file):
  """Get the directory a model is saved."""
  return os.path.join(base_load_dir, 'test_model')


def get_ckpts_from_ckpt_file(logdir):
  if gfile.Exists(logdir + '/checkpoint'):
    with gfile.Open(logdir + '/checkpoint', 'r') as f:
      ckpts = []
      for line in f:
        ckpts.append(line.strip().split()[1][1:-1])
      return ckpts
  else:
    return None


def avg(nums):
  return sum(nums) / len(nums)

def map_confusion_matrix(cm):
  reverse_dict = reverse_dictionary(char_to_num_map)
  out_dict = defaultdict(dict)
  for i, prediction_counts in enumerate(cm):
    for j, count in enumerate(prediction_counts):
      out_dict[reverse_dict[i]][reverse_dict[j]] = count
  return out_dict


def print_confusion_matrix_dict(cmd):
  for char, counts in cmd.iteritems():
    print(char, ':', counts)


def get_graph(input_x, input_seqlens, model, n_symbols, num_layers, state_size,
              dropout_keep):
  """Creates the TF RNN graph."""
  embedding = tf.Variable(np.zeros((n_symbols, state_size)).astype(np.float32))
  embedded_inputs = tf.nn.embedding_lookup(embedding, input_x)

  stacked_cell, _ = get_stacked_rnn_cell(model, n_symbols, num_layers,
                                         state_size, dropout_keep)
  outputs, _ = tf.nn.dynamic_rnn(
      stacked_cell,
      embedded_inputs,
      sequence_length=input_seqlens,
      dtype=tf.float32)

  return outputs


def get_stacked_rnn_cell(model, n_symbols, num_layers, state_size,
                         dropout_keep):
  """Creates the TF ops to get stacked RNN cell."""
  rnn_cells = []
  if model == 'gru':
    cells = []
    for _ in range(num_layers - 1):
      gru_cell = tf.contrib.rnn.GRUCell(state_size)
      cells.append(gru_cell)
      rnn_cells.append(gru_cell)
    gru_cell = tf.contrib.rnn.GRUCell(state_size)
    rnn_cells.append(gru_cell)
    opw_cell = tf.contrib.rnn.OutputProjectionWrapper(gru_cell, n_symbols)
    dropout_cell = tf.contrib.rnn.DropoutWrapper(
        opw_cell, state_keep_prob=dropout_keep)
    cells.append(dropout_cell)

  else:
    cells = []
    for _ in range(num_layers - 1):
      lstm_cell = tf.contrget_masksib.rnn.LSTMCell(state_size)
      cells.append(lstm_cell)
      rnn_cells.append(lstm_cell)
    lstm_cell = tf.contrib.rnn.LSTMCell(state_size)
    rnn_cells.append(lstm_cell)
    opw_cell = tf.contrib.rnn.OutputProjectionWrapper(lstm_cell, n_symbols)
    dropout_cell = tf.contrib.rnn.DropoutWrapper(
        opw_cell, state_keep_prob=dropout_keep)
    cells.append(dropout_cell)

  stacked_cell = tf.contrib.rnn.MultiRNNCell(cells)
  return stacked_cell, rnn_cells


def get_num_symbols():
  return num_symbols


def get_metrics(input_seqlens, input_y, outputs, n_symbols):
  """Sets up the TF ops for tracking network metrics."""
  padded_length, mask, inv_mask = get_masks(input_seqlens, input_y)
  print(padded_length, mask.shape, inv_mask.shape)
  loss_fn, _, _ = get_loss_fn(input_y, outputs, inv_mask, n_symbols,
                              input_seqlens)
  tf.summary.scalar('loss', loss_fn)

  acc_ops = get_acc_ops(outputs, input_y, input_seqlens, inv_mask, n_symbols,
                        padded_length, 1)
  accuracy_fn, update_acc_count_op, update_acc_total_op = acc_ops[0:3]
  zero_acc_count, zero_acc_total = acc_ops[3:5]
  tf.summary.scalar('accuracy', accuracy_fn)

  perplexities = get_perp_tensor(input_seqlens, outputs, input_y, mask,
                                 inv_mask, n_symbols)

  global_step = tf.Variable(0, name='global_step')
  out = [
      loss_fn, accuracy_fn, update_acc_count_op, update_acc_total_op,
      zero_acc_count, zero_acc_total, perplexities, global_step
  ]
  return out


def get_loss_fn(input_y, outputs, inv_mask, n_symbols, input_seqlens):
  """Creates the TF ops to get loss metrics."""
  # makes input to one hot -1, so that one hot always outputs zeroes,
  #  making cross entropy loss 0
  padded_neg_one_input_y = tf.add(
      tf.cast(input_y, tf.int64), tf.cast(tf.multiply(inv_mask, -1), tf.int64))
  fixed_one_hot_targets = tf.one_hot(padded_neg_one_input_y, n_symbols, axis=-1)

  mean_loss_fn = tf.losses.softmax_cross_entropy(
      fixed_one_hot_targets, outputs, reduction=tf.losses.Reduction.MEAN)
  sum_loss_fn = tf.losses.softmax_cross_entropy(
      fixed_one_hot_targets, outputs, reduction=tf.losses.Reduction.SUM)
  batch_and_length_normalized_loss_fn = tf.div(sum_loss_fn,
                                               tf.cast(
                                                   tf.reduce_sum(input_seqlens),
                                                   tf.float32))
  return batch_and_length_normalized_loss_fn, sum_loss_fn, mean_loss_fn


def get_masks(input_seqlens, input_y):
  # set up a mask around the end of the each variable length sequence
  # padded_length = tf.shape(outputs)[-2]
  padded_length = tf.shape(input_y)[-1]
  mask = tf.cast(
      tf.sequence_mask(input_seqlens, maxlen=padded_length), tf.float32)
  inv_mask = tf.cast(
      tf.logical_not(tf.sequence_mask(input_seqlens, maxlen=padded_length)),
      tf.float32)
  return padded_length, mask, inv_mask


def get_confusion_matrix(outputs, input_y, input_seqlens, inv_mask, n_symbols,
                         padded_length, batch_size):
  """Creates the TF ops to get confusion metrics."""
  output_fix = tf.pad(
      tf.expand_dims(inv_mask, axis=-1),
      tf.constant([[0, 0], [0, 0], [0, n_symbols - 1]], dtype=tf.int32))
  fixed_outputs = tf.add(outputs, output_fix)
  predictions = tf.argmax(fixed_outputs, 2)

  reshaped_predictions = tf.reshape(predictions, shape=[-1])
  reshaped_inputy = tf.reshape(input_y, shape=[-1])
  total_num_padding_chars = tf.multiply(
      padded_length, batch_size) - tf.reduce_sum(input_seqlens)
  new_confusion_matrix = tf.confusion_matrix(
      reshaped_inputy, reshaped_predictions, num_classes=n_symbols)
  padding_len_tensor = tf.expand_dims(
      tf.expand_dims(total_num_padding_chars, 0), 0)
  new_confusion_matrix_wo_padding = tf.subtract(
      new_confusion_matrix,
      tf.pad(padding_len_tensor,
             tf.constant([[0, n_symbols - 1], [0, n_symbols - 1]])))

  confusion_matrix = tf.Variable(
      tf.zeros((n_symbols, n_symbols), dtype=tf.int32),
      name='local_confusion_matrix',
      collections=[tf.GraphKeys.LOCAL_VARIABLES])

  update_cm_op = confusion_matrix.assign_add(new_confusion_matrix_wo_padding)

  return tf.identity(confusion_matrix), update_cm_op, new_confusion_matrix[
      n_symbols - 1, n_symbols - 1]


def get_acc_ops(outputs, input_y, input_seqlens, inv_mask, n_symbols,
                padded_length, batch_size):
  """Creates the TF ops to get accuracy metrics."""
  # set the output of the network (at end of each sequence) to [1,0,0,0...] to
  # match one hot target (target is '0' (default padding), so target softmax
  # is [1,0,0,...])
  output_fix = tf.pad(
      tf.expand_dims(inv_mask, axis=-1),
      tf.constant([[0, 0], [0, 0], [0, n_symbols - 1]], dtype=tf.int32))
  fixed_outputs = tf.add(outputs, output_fix)

  predictions = tf.argmax(fixed_outputs, 2)
  acc_count = tf.Variable(
      0, name='local_count', collections=[tf.GraphKeys.LOCAL_VARIABLES])
  acc_total = tf.Variable(
      0, name='local_total', collections=[tf.GraphKeys.LOCAL_VARIABLES])
  num_correct = tf.reduce_sum(
      tf.reduce_sum(
          tf.cast(tf.equal(tf.cast(input_y, tf.int64), predictions), tf.int32)))
  total_num_padding_chars = tf.multiply(
      padded_length, batch_size) - tf.reduce_sum(input_seqlens)
  update_acc_count_op = acc_count.assign_add(num_correct -
                                             total_num_padding_chars)
  update_acc_total_op = acc_total.assign_add(tf.reduce_sum(input_seqlens))

  zac = acc_count.assign(tf.constant(0))
  zat = acc_total.assign(tf.constant(0))

  accuracy_fn = tf.divide(acc_count, acc_total)

  return accuracy_fn, update_acc_count_op, update_acc_total_op, zac, zat


def get_var_by_name(name, collection):
  return [v for v in collection if v.name == name][0]


def get_vars_by_starting_pattern(pattern, collection):
  return [v for v in collection if v.name.startswith(pattern)]


def get_vars_by_ending_pattern(pattern, collection):
  return [v for v in collection if v.name.endswith(pattern)]


def get_vars_by_matching_pattern(pattern, collection):
  return [v for v in collection if pattern in v.name]


def print_var_collection(vs, shape=False):
  for v in vs:
    if shape:
      print(v.name, v.shape)
    else:
      print(v.name)


def print_all_tf_vars():
  print('\nlocal vars\n')
  print_var_collection(tf.local_variables())
  print('\nglobal vars\n')
  print_var_collection(tf.global_variables())
  print()


def get_input_producers(train_files, batch_size, num_epochs):
  _, input_seq = get_sequences(train_files, batch_size, num_epochs)
  return input_x, input_y, input_seqlens



def get_sequences(train_filenames, batch_size, num_epochs):
  """Reads the input from files and feed to network."""
  filename_queue = tf.train.string_input_producer(train_filenames, num_epochs=num_epochs)
  reader = tf.TFRecordReader()
  key, serialized_sequence_example = reader.read(filename_queue)
  _, features = tf.parse_single_sequence_example(
      serialized_sequence_example,
      sequence_features={
          'angle': tf.FixedLenSequenceFeature([], dtype=tf.int64),
          'speed': tf.FixedLenSequenceFeature([], dtype=tf.int64)
          'length': tf.FixedLenSequenceFeature([], dtype=tf.int64)
      },)

  angle = features['angle']
  speed = features['speed']
  length = tf.cast(features['length'], tf.int32)
  batched_angle, batched_speed, batched_len = tf.batch(
      [angle, speed, length],
      batch_size,
      dynamic_pad=True,
      allow_smaller_final_batch=True)

  out_seqlen = tf.squeeze(batched_len)
  out_seqlen.set_shape((batch_size,))

  return key, batched_seq, out_seqlen

