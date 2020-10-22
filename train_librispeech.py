import os
import sys
import time
import numpy as np
import tensorflow as tf
from datetime import datetime

from models import CNNBLSTMCalssifier
from librispeech_dataset import train_generator, test_generator

# some super parameters
reuse_log = True
BATCH_SIZE = 64
# BATCH_SIZE = 16
STEPS = int(5e5)
LEARNING_RATE = 0.3
if reuse_log:
    STARTED_DATESTRING = '2020-09-21T12-38-11'
else:
    STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
MAX_TO_SAVE = 20
CKPT_EVERY = 1000
MFCC_DIM = 39
PPG_DIM = 347

# 现在处理的数据集，超参数配置，均是：librispeech，所以log等文件夹名字都是librispeec
logdir = 'librispeech_log_dir'
model_dir = 'librispeech_ckpt_model_dir'
restore_dir = 'librispeech_ckpt_model_dir'


def save_model(saver, sess, logdir, step):
    model_name = 'librispeechASR.ckpt'
    ckpt_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, ckpt_path, global_step=step)
    print('Done')


def load_model(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir), end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def main():
    train_dir = os.path.join(logdir, STARTED_DATESTRING, 'train')
    dev_dir = os.path.join(logdir, STARTED_DATESTRING, 'dev')

    # dataset
    train_set = tf.data.Dataset.from_generator(train_generator,
                                               output_types=(
                                                   tf.float32, tf.float32, tf.int32),
                                               output_shapes=(
                                                   [None, MFCC_DIM], [None, PPG_DIM], []))
    #padding train data
    train_set = train_set.padded_batch(BATCH_SIZE,
                                       padded_shapes=([None, MFCC_DIM],
                                                      [None, PPG_DIM],
                                                      [])).repeat()

    train_iterator = train_set.make_initializable_iterator()

    test_set = tf.data.Dataset.from_generator(test_generator,
                                              output_types=(
                                                  tf.float32, tf.float32, tf.int32),
                                              output_shapes=(
                                                  [None, MFCC_DIM], [None, PPG_DIM], []))

    #设置repeat()，在get_next循环中，如果越界了就自动循环。不计上限
    test_set = test_set.padded_batch(BATCH_SIZE,
                                     padded_shapes=([None, MFCC_DIM],
                                                    [None, PPG_DIM],
                                                    [])).repeat()
    test_iterator = test_set.make_initializable_iterator()

    #创建一个handle占位符，在sess.run该handle迭代器的next()时，可以送入一个feed_dict 代表handle占位符，从而调用对应的迭代器
    dataset_handle = tf.placeholder(tf.string, shape=[])
    dataset_iter = tf.data.Iterator.from_string_handle(
        dataset_handle,
        train_set.output_types,
        train_set.output_shapes
    )
    batch_data = dataset_iter.get_next()

    # 调用模型
    classifier = CNNBLSTMCalssifier(out_dims=PPG_DIM, n_cnn=3, cnn_hidden=256,
                                    cnn_kernel=3, n_blstm=2, lstm_hidden=128)
    # 模型output
    results_dict = classifier(batch_data[0], batch_data[1], batch_data[2])
    #inputs labels lengths
    #results_dict['logits']= np.zeros([10])
    predicted = tf.nn.softmax(results_dict['logits'])
    mask = tf.sequence_mask(batch_data[2], dtype=tf.float32)#batch[2]是（None,)，是每条数据的MFCC数目,需要这个的原因是会填充成最长的MFCC长度。mask的维度是(None,max(batch[2]))



    #batch_data[2]是MFCC数组的长度，MFCC有多少是不一定的。
    accuracy = tf.reduce_sum(
        tf.cast(#bool转float
            tf.equal(tf.argmax(predicted, axis=-1),#比较每一行的最大元素
                     tf.argmax(batch_data[1], axis=-1)),
            tf.float32) * mask#乘上mask，是因为所有数据都被填充为最多mfcc的维度了。所以填充部分一定都是相等的，于是需要将其mask掉。
    ) / tf.reduce_sum(tf.cast(batch_data[2], dtype=tf.float32))

    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('predicted',
                     tf.expand_dims(
                         tf.transpose(predicted, [0, 2, 1]),
                         axis=-1), max_outputs=1)
    tf.summary.image('groundtruth',
                     tf.expand_dims(
                         tf.cast(
                             tf.transpose(batch_data[1], [0, 2, 1]),
                             tf.float32),
                         axis=-1), max_outputs=1)

    loss = results_dict['cross_entropy']
    learning_rate_pl = tf.placeholder(tf.float32, None, 'learning_rate')
    tf.summary.scalar('cross_entropy', loss)
    tf.summary.scalar('learning_rate', learning_rate_pl)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate_pl)
    optim = optimizer.minimize(loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optim = tf.group([optim, update_ops])

    # Set up logging for TensorBoard.
    train_writer = tf.summary.FileWriter(train_dir)
    train_writer.add_graph(tf.get_default_graph())
    dev_writer = tf.summary.FileWriter(dev_dir)
    summaries = tf.summary.merge_all()
    #设置将所有的summary保存 run这个会将所有的summary更新
    saver = tf.train.Saver(max_to_keep=MAX_TO_SAVE)

    # set up session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run([train_iterator.initializer, test_iterator.initializer])
    train_handle, test_handle = sess.run([train_iterator.string_handle(),
                                         test_iterator.string_handle()])
    sess.run(init)
    # try to load saved model
    try:
        saved_global_step = load_model(saver, sess, restore_dir)
        if saved_global_step is None:
            saved_global_step = -1
    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    last_saved_step = saved_global_step
    step = None
    try:
        for step in range(saved_global_step + 1, STEPS):
            if step <= int(4e5):
                lr = LEARNING_RATE
            elif step <= int(6e5):
                lr = 0.5 * LEARNING_RATE
            elif step <= int(8e5):
                lr = 0.25 * LEARNING_RATE
            else:
                lr = 0.125 * LEARNING_RATE
            start_time = time.time()
            if step % CKPT_EVERY == 0:
                summary, loss_value = sess.run([summaries, loss],
                                               feed_dict={dataset_handle: test_handle,
                                                          learning_rate_pl: lr})
                dev_writer.add_summary(summary, step)
                duration = time.time() - start_time
                print('step {:d} - eval loss = {:.3f}, ({:.3f} sec/step)'
                      .format(step, loss_value, duration))
                save_model(saver, sess, model_dir, step)
                last_saved_step = step
            else:
                summary, loss_value, _ = sess.run([summaries, loss, optim],
                                                  feed_dict={dataset_handle: train_handle,
                                                             learning_rate_pl: lr})
                train_writer.add_summary(summary, step)
                if step % 10 == 0:
                    duration = time.time() - start_time
                    print('step {:d} - training loss = {:.3f}, ({:.3f} sec/step)'
                          .format(step, loss_value, duration))
    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save_model(saver, sess, model_dir, step)
    sess.close()


if __name__ == '__main__':
    main()
