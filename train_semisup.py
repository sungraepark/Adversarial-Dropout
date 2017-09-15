import time

import numpy
import tensorflow as tf
import math
import os
import layers as L

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('device', '/gpu:0', "device")
tf.app.flags.DEFINE_string('dataset', 'cifar10', "{cifar10, svhn}")
tf.app.flags.DEFINE_string('log_dir', "", "log_dir")
tf.app.flags.DEFINE_integer('seed', 5, "initial random seed")

tf.app.flags.DEFINE_integer('batch_size', 32, "the number of examples in a batch")
tf.app.flags.DEFINE_integer('ul_batch_size', 128, "the number of unlabeled examples in a batch")
tf.app.flags.DEFINE_integer('eval_batch_size', 100, "the number of eval examples in a batch")
tf.app.flags.DEFINE_integer('eval_freq', 1, "")
tf.app.flags.DEFINE_integer('num_epochs', 300, "the number of epochs for training")
tf.app.flags.DEFINE_integer('epoch_decay_start', 250, "epoch of starting learning rate decay")
tf.app.flags.DEFINE_integer('num_iter_per_epoch', 400, "the number of updates per epoch")
tf.app.flags.DEFINE_float('learning_rate', 0.003, "initial leanring rate")
tf.app.flags.DEFINE_float('mom1', 0.9, "initial momentum rate")
tf.app.flags.DEFINE_float('mom2', 0.5, "momentum rate after epoch_decay_start")

tf.app.flags.DEFINE_string('method', 'VAdD-KL', "{VAdD-KL, VAdD-QE, VAT, Pi, VAT+VAdD-KL, VAT+VAdD-QE}")

tf.app.flags.DEFINE_float('delta', 0.1, "lower bound of mask difference")
tf.app.flags.DEFINE_float('lamb_max', 100.0, "coefficient of adversarial dropout regularizer")
tf.app.flags.DEFINE_integer('rampup_length', 80, "rampup_length_for_lamb")
tf.app.flags.DEFINE_integer('rampdown_length', 50, "rampup_length_for_lamb")

import adt

lamb_max = FLAGS.lamb_max
print lamb_max

def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0
    
def rampdown(epoch, rampdown_length, total_epoch):
    if epoch >= (total_epoch - rampdown_length):
        ep = (epoch - (total_epoch - rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / rampdown_length)
    else:
        return 1.0 
    
if FLAGS.dataset == 'cifar10':
    from cifar10_semisup import one_transformed_input, two_transformed_input, unlabeled_two_transformed_input
    NUM_EVAL_EXAMPLES = 10000
elif FLAGS.dataset == 'svhn':
    from svhn_semisup import one_transformed_input, two_transformed_input, unlabeled_two_transformed_input
    NUM_EVAL_EXAMPLES = 26032
else: 
    raise NotImplementedError



def build_training_graph(x_1, x_2, y, ul_x_1, ul_x_2, lr, mom, lamb):
    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False,
    )
    logit = adt.forward(x_1, update_batch_stats=True)
    nll_loss = L.ce_loss(logit, y)
    
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        
        if FLAGS.method == 'VAdD-KL':
    
            ul_logit = adt.forward(ul_x_1)
            ul_adt_logit = adt.forward_adv_drop(ul_x_2, ul_logit, FLAGS.delta, is_training=True, mode=FLAGS.method)
            
            additional_loss = L.kl_divergence_with_logit(ul_logit, ul_adt_logit) #*4.0
            ent_loss = L.entropy_y_x(ul_logit)
    
            loss = nll_loss + lamb * additional_loss + ent_loss
    
        elif FLAGS.method == 'VAT+VAdD-KL':
    
            ul_logit = adt.forward(ul_x_1)
            ul_adt_logit = adt.forward_adv_drop(ul_x_2, ul_logit, FLAGS.delta, is_training=True, mode='VAdD-KL')
            
            additional_loss = L.kl_divergence_with_logit(ul_logit, ul_adt_logit) #*4.0
            ent_loss = L.entropy_y_x(ul_logit)
            vat_loss = adt.virtual_adversarial_loss(ul_x_1, ul_logit)
            
            loss = nll_loss + lamb * additional_loss + vat_loss + ent_loss
            
        elif FLAGS.method == 'VAdD-QE':
            ul_logit = adt.forward(ul_x_1, update_batch_stats=True)
            ul_adt_logit = adt.forward_adv_drop(ul_x_2, ul_logit, FLAGS.delta, is_training=True, update_batch_stats=True, mode=FLAGS.method)
            
            additional_loss = L.qe_loss(ul_logit, ul_adt_logit) #*4.0
            ent_loss = L.entropy_y_x(ul_logit)
            
            loss = nll_loss + lamb * additional_loss + ent_loss
        
        elif FLAGS.method == 'VAT+VAdD-QE':
            ul_logit = adt.forward(ul_x_1, update_batch_stats=True)
            ul_adt_logit = adt.forward_adv_drop(ul_x_2, ul_logit, FLAGS.delta, is_training=True, update_batch_stats=True, mode='VAdD-QE')
    
            additional_loss = L.qe_loss(ul_logit, ul_adt_logit) #*4.0
            ent_loss = L.entropy_y_x(ul_logit)
            vat_loss = adt.virtual_adversarial_loss(ul_x_1, ul_logit)
    
            loss = nll_loss + lamb * additional_loss + vat_loss + ent_loss
            
        elif FLAGS.method == 'VAT':
            
            ul_logit = adt.forward(ul_x_1, update_batch_stats=False)
            ent_loss = L.entropy_y_x(ul_logit) # + L.entropy_y_x(ul_adt_logit)
            vat_loss = adt.virtual_adversarial_loss(ul_x_1, ul_logit)
            loss = nll_loss + vat_loss + ent_loss
            
        elif FLAGS.method == 'Pi':
            
            ul_logit = adt.forward(ul_x_1, update_batch_stats=True)
            ul_adt_logit = adt.forward(ul_x_2, update_batch_stats=True)
            additional_loss = L.qe_loss(ul_logit, ul_adt_logit) #*4.0
            ent_loss = L.entropy_y_x(ul_logit)
    
            loss = nll_loss + lamb * additional_loss + ent_loss
            
        elif FLAGS.method == 'baseline':
            logit = vat.forward(x_1)
            nll_loss = L.ce_loss(logit, y)
            scope = tf.get_variable_scope()
            scope.reuse_variables()
            
            additional_loss = 0
        else:
            raise NotImplementedError

    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=mom, beta2=0.999)
        tvars = tf.trainable_variables()
        grads_and_vars = opt.compute_gradients(loss, tvars)
        train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
    return loss, train_op, global_step


def build_test_graph(x, y):
    losses = {}
    logit= adt.forward(x, is_training=False, update_batch_stats=False)
    nll_loss = L.ce_loss(logit, y)
    losses['No_NLL'] = nll_loss
    acc = L.accuracy(logit, y)
    losses['No_Acc'] = acc
    return losses


def main(_):
    print(FLAGS.lamb_max, FLAGS.delta, FLAGS.epsilon, FLAGS.top_bn, FLAGS.sigma)
    print lamb_max
    numpy.random.seed(seed=FLAGS.seed)
    tf.set_random_seed(numpy.random.randint(1234)) 
    with tf.Graph().as_default() as g:
        with tf.device("/cpu:0"):
            images_1, images_2, labels = two_transformed_input(batch_size=FLAGS.batch_size,train=True, shuffle=True)
            ul_images_1, ul_images_2 = unlabeled_two_transformed_input(batch_size=FLAGS.ul_batch_size, shuffle=True)
            images_eval_test, labels_eval_test = one_transformed_input(batch_size=FLAGS.eval_batch_size, train=False, shuffle=True)

        with tf.device(FLAGS.device):
            lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
            mom = tf.placeholder(tf.float32, shape=[], name="momentum")
            lamb = tf.placeholder(tf.float32, shape=[], name="regular_coef")
            
            with tf.variable_scope("CNN") as scope:
                # Build training graph
                loss, train_op, global_step = build_training_graph(images_1, images_2, labels, ul_images_1, ul_images_2, lr, mom, lamb)
                scope.reuse_variables()
                # Build eval graph
                losses_eval_test = build_test_graph(images_eval_test, labels_eval_test)

            init_op = tf.global_variables_initializer()

        if not FLAGS.log_dir:
            logdir = None
            writer_test = None
        else:
            logdir = FLAGS.log_dir
            if os.path.isdir(logdir)==False:
                os.makedirs(logdir)
            writer_test = tf.summary.FileWriter(FLAGS.log_dir + "/test", g)

        saver = tf.train.Saver(tf.global_variables())
        sv = tf.train.Supervisor(
            is_chief=True,
            logdir=logdir,
            init_op=init_op,
            init_feed_dict={lr: FLAGS.learning_rate, mom: FLAGS.mom1},
            saver=saver,
            global_step=global_step,
            summary_op=None,
            summary_writer=None,
            save_model_secs=150, recovery_wait_secs=0)

        print("Training...")
        with sv.managed_session() as sess:
            for ep in range(FLAGS.num_epochs):
                if sv.should_stop():
                    break
                
                rampup_value = rampup(ep, FLAGS.rampup_length)
                rampdown_value = rampdown(ep, FLAGS.rampdown_length, FLAGS.num_epochs)
            
                rampup_lamb = lamb_max*rampup_value
                learning_rate = FLAGS.learning_rate * rampdown_value * rampup_value
                adam_beta1 = rampdown_value * FLAGS.mom1 + (1.0 - rampdown_value) * FLAGS.mom2
                
                feed_dict = {lr: learning_rate, mom: adam_beta1, lamb:rampup_lamb}

                sum_loss = 0
                start = time.time()
                for i in range(FLAGS.num_iter_per_epoch):
                    _, batch_loss, _ = sess.run([train_op, loss, global_step],
                                                feed_dict=feed_dict)
                    sum_loss += batch_loss
                end = time.time()
                
                if (ep + 1) % FLAGS.eval_freq == 0 or ep + 1 == FLAGS.num_epochs:
                    
                    # Eval on test data
                    act_values_dict = {}
                    for key, _ in losses_eval_test.iteritems():
                        act_values_dict[key] = 0
                    n_iter_per_epoch = NUM_EVAL_EXAMPLES / FLAGS.eval_batch_size
                    for i in range(n_iter_per_epoch):
                        values = losses_eval_test.values()
                        act_values = sess.run(values)
                        for key, value in zip(act_values_dict.keys(), act_values):
                            act_values_dict[key] += value
                    summary = tf.Summary()
                    current_global_step = sess.run(global_step)
                    for key, value in act_values_dict.iteritems():
                        summary.value.add(tag=key, simple_value=value / n_iter_per_epoch)
                    if writer_test is not None:
                        writer_test.add_summary(summary, current_global_step)
                    
                    cur_error = act_values_dict["No_Acc"]/n_iter_per_epoch

                    print("Epoch:", ep, "CE_loss_train:", sum_loss / FLAGS.num_iter_per_epoch, "test error", cur_error, "elapsed_time:", end - start)
                else:
                    print("Epoch:", ep, "CE_loss_train:", sum_loss / FLAGS.num_iter_per_epoch, "elapsed_time:", end - start)

            #saver.save(sess, sv.save_path, global_step=global_step)
        sv.stop()


if __name__ == "__main__":
    tf.app.run()