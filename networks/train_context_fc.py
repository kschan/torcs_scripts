"""Trains amazing RNN language models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import tensorflow as tf
import os
from helpers import get_input_file_paths
from helpers import get_load_path
from helpers import get_dataset_counts
from helpers import get_graph
from helpers import get_input_producers
from helpers import get_metrics
from helpers import print_var_collection
from helpers import mkdirp

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_config', "model_config.txt", 'File defining the network.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_float('starting_lr', 1e-3, 'Learning rate.')
flags.DEFINE_float('lr_decay', 0.98, 'Learning rate.')
flags.DEFINE_integer('num_epochs', 200, 'Number of epochs.')

flags.DEFINE_string('base_save_dir', './model_checkpoints/',
                    'Model save base folder')
flags.DEFINE_string('datasets_dir',
                    '../logs/',
                    'Directory with data: ./[dataset_name]/*.npy.')


def main(argv):
  del argv

  batch_size = FLAGS.batch_size
  num_epochs = FLAGS.num_epochs
  starting_lr = FLAGS.starting_lr

  save_dir = get_load_path(FLAGS.base_save_dir, FLAGS.model_config)
  mkdirp(save_dir)
  print('Using model save path:', save_dir)

  train_files = get_input_file_paths(FLAGS.datasets_dir, "train")
  print('Using input files:', train_files)

  tf.reset_default_graph()
  with tf.Graph().as_default():
    input_x, input_y, input_seqlens = get_input_producers(train_files, batch_size, num_epochs)

    sys.exit(0)
    outputs = get_graph(input_x, input_seqlens, FLAGS.model, num_symbols,
                        num_layers, state_size, dropout)
    metric = get_lm_metrics(input_seqlens, input_y, outputs, num_symbols)
    loss_fn, accuracy_fn, update_acc_count_op, update_acc_total_op = metric[0:4]
    zero_acc_count, zero_acc_total, global_step = metric[4:7]
    lr = tf.train.exponential_decay(FLAGS.starting_lr, global_step, 100000,
                                    FLAGS.lr_decay)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss_fn, global_step=global_step)

    print_var_collection(tf.global_variables(), True)

    sv = tf.Supervisor(
        is_chief=True,
        logdir=save_dir,
        init_op=tf.Supervisor.USE_DEFAULT,
        local_init_op=tf.Supervisor.USE_DEFAULT,
        ready_for_local_init_op=tf.Supervisor.USE_DEFAULT,
        global_step=global_step,
        summary_op=tf.Supervisor.USE_DEFAULT,
        saver=tf.train.Saver(var_list=tf.global_variables(), max_to_keep=6000),
        save_summaries_secs=30,
        save_model_secs=90)

    num_chars, losses = [], []
    num_steps_in_epoch = int(get_dataset_counts()['train'][FLAGS.dataset] / batch_size)
    num_steps_training = num_epochs * num_steps_in_epoch
    print('Number of steps in epoch:', num_steps_in_epoch,
          'Num steps training:', num_steps_training)

    try:
      with sv.managed_session('local') as sess:
        sv.StartQueueRunners(sess)
        for _ in range(num_steps_training):
          _, step, loss, _, _, acc, seq_lens = sess.run([
              train_op, global_step, loss_fn, update_acc_count_op,
              update_acc_total_op, accuracy_fn, input_seqlens
          ])

          num_chars.extend(seq_lens.tolist())
          losses.append(loss)

          avg_loss = sum(losses) / len(losses)

          out = '(step,acc,loss,weights): (%d, %f, %f, %f)' % (step, acc,
                                                                    avg_loss)

          if step % 10 == 0:
            print(out)

          if step % num_steps_in_epoch == 0:
            with open(save_dir + '/train_results.txt', 'a') as f:
              f.write(out + '\n')
            num_chars, losses = [], []
            sess.run([zero_acc_total, zero_acc_count])

          if sv.coord.should_stop():
            break

        with open(save_dir + '/train_results.txt', 'a') as f:
          f.write('done\n')

    except tf.errors.OutOfRangeError:
      print('Done with all epochs')
    except tf.errors.AbortedError:
      print('Aborted error!')
    sv.Stop()


if __name__ == '__main__':
  tf.app.run(main)
