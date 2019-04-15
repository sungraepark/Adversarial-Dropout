import tensorflow as tf
import numpy
import sys, os

import layers as L
import cnn as CNN

FLAGS = tf.app.flags.FLAGS

def logit(x, masks=None, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):
    logits, _ =  CNN.logit(x, masks, is_training=is_training,
                     update_batch_stats=update_batch_stats,
                     stochastic=stochastic,
                     seed=seed)
    return logits

def forward(x, is_training=True, update_batch_stats=True, seed=1234):
    if is_training:
        return logit(x, is_training=True,
                     update_batch_stats=update_batch_stats,
                     stochastic=True, seed=seed)
    else:
        return logit(x, is_training=False,
                     update_batch_stats=update_batch_stats,
                     stochastic=False, seed=seed)


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

def generate_virtual_adversarial_dropout_mask(x, logit, is_training=True):
    
    logit_m, init_mask =  CNN.logit(x, None, is_training=True, update_batch_stats=False,
                                    stochastic=True, seed=1234)
    dist = L.kl_divergence_with_logit(logit_m, logit)
    mask_grad = tf.stop_gradient(tf.gradients(dist, [init_mask], aggregation_method=2)[0])
    return flipping_algorithm(init_mask, mask_grad)

    
def virtual_adversarial_dropout_loss(x, logit, is_training=True, name="vadt_loss"):
    adv_mask = generate_virtual_adversarial_dropout_mask(x, logit, is_training=is_training)
    logit_p = logit
    logit_m, _ = CNN.logit(x, adv_mask, is_training=True, update_batch_stats=True,
                                    stochastic=True, seed=1234)

    loss = L.kl_divergence_with_logit(logit_p, logit_m)
    return tf.identity(loss, name=name)

def flipping_algorithm(init_mask, Jacobian, name="adv_filpping"):
    
    dim = tf.reduce_prod(tf.shape(init_mask)[1:])
    change_limit = int(CNN.layer_sizes[4]*FLAGS.delta)
    changed_mask = init_mask
    
    if change_limit != 0 :   
        dir = tf.reshape(Jacobian, [-1, dim])
        
        # mask (init_mask=1->m=1), (init_mask=0->m=-1)
        m = init_mask
        m = 2.0*m - tf.ones_like(m)
        
        # sign of Jacobian  (J>0 -> s=1), (J<0 -> s= -1)
        s = tf.cast( tf.greater(dir, float(0.0)), tf.float32)
        s = 2.0*s - tf.ones_like(s)                    
        
        # remain (J>0, m=-1) and (J<0, m=1), which are candidates to be changed
        change_candidate = tf.cast( tf.less( s*m, float(0.0) ), tf.float32) # s = -1, m = 1
        ads_dL_dm = tf.abs(dir)
        
        # ordering abs_Jacobian for candidates
        # the maximum number of the changes is "change_limit"
        # draw top_k elements ( if the top k element is 0, the number of the changes is less than "change_limit" ) 
        left_values = change_candidate*ads_dL_dm
        with tf.device("/cpu:0"):
            min_left_values = tf.nn.top_k(left_values, change_limit+1)[0][:,-1]    
        change_target = tf.cast(  tf.greater(left_values, tf.expand_dims(min_left_values, -1) ), tf.float32)
        
        # changed mask with change_target
        changed_mask = (m - 2.0*m*change_target + tf.ones_like(m))*0.5 
        
        # normalization
        sum_mask = tf.cast( tf.reduce_sum(changed_mask, axis=1, keep_dims=True), tf.float32)
        changed_mask = tf.div(changed_mask,sum_mask)*tf.to_float(dim)
        changed_mask = tf.reshape(changed_mask, tf.shape(init_mask))
    
    return changed_mask


if __name__ == "__main__":
    tf.app.run()