import tensorflow as tf
import numpy
import sys, os

import layers as L
import cnn_with_adv_drop as CNN

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('epsilon', 8.0, "norm length for (virtual) adversarial training ")
tf.app.flags.DEFINE_integer('num_power_iterations', 1, "the number of power iterations")
tf.app.flags.DEFINE_float('xi', 1e-6, "small constant for finite difference")


def logit(x, masks=None, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):
    return CNN.logit(x, is_training=is_training,
                     update_batch_stats=update_batch_stats,
                     stochastic=stochastic,
                     seed=seed)


def forward(x, is_training=True, update_batch_stats=True, seed=1234):
    if is_training:
        return logit(x, is_training=True,
                     update_batch_stats=update_batch_stats,
                     stochastic=True, seed=seed)
    else:
        return logit(x, is_training=False,
                     update_batch_stats=update_batch_stats,
                     stochastic=False, seed=seed)

def forward_adv_drop(x, y, delta,  is_training=True, update_batch_stats=True, mode='adt', name="adt_loss"):
    if mode=='SAdD':
        return CNN.logit_adversarial_dropout(x, y, delta, is_training=is_training, update_batch_stats=update_batch_stats)
    elif mode=='VAdD-KL':
        return CNN.logit_adversarial_dropout(x, y, delta, is_training=is_training, update_batch_stats=update_batch_stats, _loss_fuc_=L.kl_divergence_with_logit)
    elif mode=='VAdD-QE':
        return CNN.logit_adversarial_dropout(x, y, delta, is_training=is_training, update_batch_stats=update_batch_stats, _loss_fuc_=L.qe_loss)
    else:
        return None

def get_normalized_vector(d):
    d /= (1e-12 + tf.reduce_max(tf.abs(d), range(1, len(d.get_shape())), keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), range(1, len(d.get_shape())), keep_dims=True))
    return d


def generate_virtual_adversarial_perturbation(x, logit, is_training=True):
    d = tf.random_normal(shape=tf.shape(x))

    for _ in range(FLAGS.num_power_iterations):
        d = FLAGS.xi * get_normalized_vector(d)
        logit_p = logit
        logit_m = forward(x + d, update_batch_stats=False, is_training=is_training)
        dist = L.kl_divergence_with_logit(logit_p, logit_m)
        grad = tf.gradients(dist, [d], aggregation_method=2)[0]
        d = tf.stop_gradient(grad)

    return FLAGS.epsilon * get_normalized_vector(d)


def virtual_adversarial_loss(x, logit, is_training=True, name="vat_loss"):
    r_vadv = generate_virtual_adversarial_perturbation(x, logit, is_training=is_training)
    logit = tf.stop_gradient(logit)
    logit_p = logit
    logit_m = forward(x + r_vadv, update_batch_stats=False, is_training=is_training)
    loss = L.kl_divergence_with_logit(logit_p, logit_m)
    return tf.identity(loss, name=name)

if __name__ == "__main__":
    tf.app.run()