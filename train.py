import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
import numpy as np
import collections
import os
import re
import time
import yelp_reader

import pdb

"""
https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf

1) word to sent encodings w/ attention

h_i_t --> hidden rep from bi-directional GRU (concat fw and bw)
u_i_t --> act(W_w * h_i_t + b_w)
u_w --> word context vector (randomly initialized)
alpha_i_t --> softmax(u_i_t^T u_w)
sent_rep/s_i --> sigma(alpha_i_t * h_i_t)

2) sent to doc encodings w/ attention

h_i --> hidden rep from bi-directional GRU (concat fw and bw)
u_i --> act(W_s * h_i + b_s)
u_s --> sent context vector (randomly initialized)
alpha_i --> softmax(u_i^T * u_s)
doc_rep/v --> sigma(alpha_i * h_i)

3) classify

labels distro --> softmax(W_c * v + b_c)

"""

vocab_size = 10000
embed_dim = 100
hidden_layer_dim = 100
attn_dim = 200
sent_len = 30
doc_len = 5
n_labels = 5
batch_size = 64
n_epochs = 30
check_in = 200
log_vars = False
checkpoints = "./checkpoints/"
assets = "./assets/"
default_init = tf.random_normal_initializer(stddev=1.0)

tf.logging.set_verbosity(tf.logging.INFO)

def variable_summaries(variables=[]):
    for v in variables:
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(v)
            tf.summary.scalar('mean', mean)
            tf.summary.histogram('histogram', v)

def inspect_gru_vars(sess):
    ret = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gru_word_to_sent')[0])
    ret1 = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gru_sent_to_doc')[0])
    print(sess.run(tf.reduce_mean(tf.reduce_sum(ret, axis=1))))
    print(sess.run(tf.reduce_mean(tf.reduce_sum(ret2, axis=1))))

global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                trainable=False)

with tf.variable_scope('embeddings'):
    embeddings = tf.get_variable("embed", [vocab_size, embed_dim], dtype=tf.float32)

""" Note: *2 because concating bidrectional bw and fw """
with tf.variable_scope("word_to_sent", initializer=default_init):
    W_w = tf.get_variable("W_w", [hidden_layer_dim*2, attn_dim])
    b_w = tf.get_variable("b_w", [attn_dim])
    u_w = tf.get_variable('u_w', [batch_size, sent_len, attn_dim])

with tf.variable_scope("sent_to_doc", initializer=default_init):
    W_s = tf.get_variable("W_s", [hidden_layer_dim*2, attn_dim])
    b_s = tf.get_variable("b_s", [attn_dim])
    u_s = tf.get_variable('u_s', [batch_size, doc_len, attn_dim])

with tf.variable_scope("classify", initializer=default_init):
    W_c = tf.get_variable("W_c", [attn_dim, n_labels])
    b_c = tf.get_variable("b_c", [n_labels])

variable_summaries([W_w, b_w, u_w, W_s, b_s, u_s, W_c, b_c])
merged = tf.summary.merge_all()

x = tf.placeholder(tf.int32, [None, doc_len, sent_len])
x_reshaped = tf.transpose(x, [1, 0, 2])
labels = tf.placeholder(tf.int32, [None, n_labels])

embedded_x = tf.nn.embedding_lookup(embeddings, x_reshaped)

def gather_sent_embeddings(i, t_array, t_array_attn):

    sent = tf.gather(embedded_x, i)

    with tf.variable_scope("gru_word_to_sent"):
        outputs_h_i_t, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
            MultiRNNCell([GRUCell(hidden_layer_dim)]),
            MultiRNNCell([GRUCell(hidden_layer_dim)]),
            tf.unstack(tf.transpose(sent, [1, 0, 2])),
            dtype=tf.float32,
            sequence_length=[sent_len]*batch_size)

    #h_i_t_full = tf.stack(axis=1, values=outputs_h_i_t)
    h_i_t_full = tf.transpose(outputs_h_i_t, [1,0,2])

    h_i_t = tf.reshape(h_i_t_full, [-1, hidden_layer_dim*2])

    u_i_t = tf.reshape(
        tf.tanh(tf.matmul(h_i_t, W_w) + b_w), [batch_size, sent_len, attn_dim])

    alpha_i_t = tf.expand_dims(
        tf.nn.softmax(tf.reduce_sum(u_w * u_i_t, axis=-1)), axis=-1)

    s_i = tf.reduce_sum(h_i_t_full * alpha_i_t, axis=1)

    t_array = t_array.write(i, s_i)
    t_array_attn = t_array_attn.write(i, alpha_i_t)
    return i+1, t_array, t_array_attn


t_array = tf.TensorArray(tf.float32, size=doc_len, dynamic_size=True, clear_after_read=False,
                        infer_shape=False)
t_array_attn = tf.TensorArray(tf.float32, size=doc_len, dynamic_size=True, clear_after_read=False,
                                infer_shape=False)

_, sent_reps_ta, word_attn_ta = tf.while_loop(
                    cond=lambda i, _1, _2: i < doc_len,
                    body=gather_sent_embeddings,
                    loop_vars=(tf.constant(0, dtype=tf.int32), t_array, t_array_attn))

alpha_i_t = tf.transpose(
        tf.reshape(word_attn_ta.stack(), [doc_len, batch_size, sent_len]), [1,0,2])

sent_reps = tf.reshape(
        sent_reps_ta.stack(), [doc_len, batch_size, attn_dim])


""" doc_len x b x attn_dim """
with tf.variable_scope("gru_sent_to_doc"):
    outputs_h_i, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
        MultiRNNCell([GRUCell(hidden_layer_dim)]),
        MultiRNNCell([GRUCell(hidden_layer_dim)]),
        tf.unstack(sent_reps),
        dtype=tf.float32,
        sequence_length=[doc_len]*batch_size)

#h_i_full = tf.stack(axis=1, values=outputs_h_i)
h_i_full = tf.transpose(outputs_h_i, [1,0,2])
h_i = tf.reshape(h_i_full, [-1, hidden_layer_dim*2])

u_i = tf.reshape(
    tf.tanh(tf.matmul(h_i, W_s) + b_s), [batch_size, doc_len, attn_dim])

alpha_i = tf.expand_dims(
    tf.nn.softmax(tf.reduce_sum(u_s * u_i, axis=-1)), axis=-1)

v = tf.reduce_sum(h_i_full * alpha_i, axis=1)

logits = tf.nn.softmax(tf.matmul(v, W_c) + b_c)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), \
    tf.argmax(labels, 1)), tf.float32))

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

opt = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-8)
grads = opt.compute_gradients(loss)
train_op = opt.apply_gradients(grads, global_step=global_step)


def train():

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        sumwriter = tf.summary.FileWriter('./summaries/', sess.graph)

        saver = save_or_restore(sess)

        """ TODO add queue for all files training """
        x_input, y, _ = yelp_reader.prepare_data(assets + "file_00.json", tiny=False)

        start_time = time.time()
        epoch_acc = []; epoch_loss = []
        train_acc = []; train_loss = []
        val_acc = []; val_loss = []

        n_steps = len(x_input) // batch_size

        for epoch_i in range(n_epochs):

            for step_i in range(n_steps):

                idx_step = step_i*batch_size
                x_batch = x_input[idx_step:idx_step+batch_size]
                y_batch = y[idx_step:idx_step+batch_size]

                fetches = [accuracy, loss, logits, train_op]
                feed = { x: x_batch, labels: y_batch }

                acc_ret, loss_ret, logits_ret, _ = sess.run(fetches, feed)

                if log_vars:
                    sumwriter.add_summary(merged_ret, epoch_i * n_steps + step_i)
                    inspect_gru_vars(sess)

                train_acc.append(acc_ret)
                train_loss.append(loss_ret)

                if step_i % check_in == 0:
                    if not os.path.exists(checkpoints): os.mkdir(checkpoints)

                    print("train step {} after {:.2f}s --> accuracy {:.6f} loss {:.6f} k_logits: {}\n" \
                        .format(sess.run(global_step), time.time() - start_time, np.mean(train_acc[-check_in:]),
                        np.mean(train_loss[-check_in:]), np.argmax(logits_ret, 1)))

                    # TODO val
                    # if epoch_i > 5 and val_acc[-1] < min(val_acc[-10:]):
                    #     continue # don't save if val dropping

                    saver.save(sess, checkpoints + "hier_attn", global_step=global_step)

            print('completed epoch {:d}\n'.format(epoch_i+1))
            epoch_acc.append(np.mean(train_acc[-n_steps:]))
            epoch_loss.append(np.mean(train_loss[-n_steps:]))


def test():
    pass


def inspect_attn(document=None):
    """ alpha_i_t and alpha_i are the softmax attn vectors """

    if not document:
        document = "I love this restaurant. It's in my neighborhood. \
                    I go here every week because the food is incredibe!"

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        saver = save_or_restore(sess)

        _, _, token_to_id = yelp_reader.prepare_data(assets + "file_00.json", tiny=False)
        x_input = yelp_reader.prepare_for_visual(document, token_to_id) * batch_size

        pdb.set_trace()

        word_attn_ret, sent_attn_ret, logits_ret = sess.run([alpha_i_t, alpha_i,
                                        logits], { x: x_input })

        word_attn = [[(y, i) for i,y in enumerate(sent)] for sent in word_attn_ret[0]]
        word_attn = [sorted(sent, key=lambda x: -x[0]) for sent in word_attn]

        sent_attn = [(x, i) for i,x in enumerate(np.squeeze(sent_attn_ret[0]))]
        sent_attn = sorted(sent_attn, key=lambda x: -x[0])

        id_to_token = {v: k for k, v in token_to_id.iteritems()}

        prediction = np.argmax(logits_ret[0])
        print(sent_attn, word_attn, prediction)



def save_or_restore(sess):
    saver = tf.train.Saver(max_to_keep=5)
    if os.path.exists(checkpoints) and os.listdir(checkpoints) != []:
        print("\nrestoring model parameters\n")
        saver.restore(sess, tf.train.latest_checkpoint(checkpoints))
    else:
        print("\ncreating model with fresh parameters.\n")
    return saver



if __name__ == "__main__":
    train()
