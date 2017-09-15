import tensorflow as tf
import numpy
import sys, os
import layers as L
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.framework import ops

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('keep_prob_hidden', 0.5, "dropout rate")
tf.app.flags.DEFINE_float('sigma', 0.0, "gaussian noise (std)")
tf.app.flags.DEFINE_float('lrelu_a', 0.1, "lrelu slope")
tf.app.flags.DEFINE_boolean('top_bn', False, "")
tf.app.flags.DEFINE_boolean('mean_only_bn', False, "")

layer_sizes = [128, 256, 512, 256, 128] #Conv-Large
#layer_sizes = [96, 192, 192, 192, 192] #Conv-Small
#layer_sizes = [64, 128, 128, 128, 128] #Conv-Small SVHN

if FLAGS.mean_only_bn:
    bn = L.mean_only_bn
else:
    bn = L.bn

def logit(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):
    h = x
    h = forward_before_adt(h, is_training=is_training, update_batch_stats=update_batch_stats, stochastic=stochastic, seed=seed)
    h = forward_after_adt(h, is_training=is_training, update_batch_stats=update_batch_stats, stochastic=stochastic, seed=seed)
    return h

def forward_before_adt(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):
    h = x

    rng = numpy.random.RandomState(seed)
    
    h = L.gl(h, std=FLAGS.sigma)
    h = L.conv(h, ksize=3, stride=1, f_in=3, f_out=layer_sizes[0], seed=rng.randint(123456), name='c1')
    h = L.lrelu(bn(h, layer_sizes[0], is_training=is_training, update_batch_stats=update_batch_stats, name='b1'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=3, stride=1, f_in=layer_sizes[0], f_out=layer_sizes[0], seed=rng.randint(123456), name='c2')
    h = L.lrelu(bn(h, layer_sizes[0], is_training=is_training, update_batch_stats=update_batch_stats, name='b2'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=3, stride=1, f_in=layer_sizes[0], f_out=layer_sizes[0], seed=rng.randint(123456), name='c3')
    h = L.lrelu(bn(h, layer_sizes[0], is_training=is_training, update_batch_stats=update_batch_stats, name='b3'), FLAGS.lrelu_a)

    h = L.max_pool(h, ksize=2, stride=2)
    
    h = tf.nn.dropout(h, keep_prob=0.5, seed=rng.randint(123456)) if stochastic else h
    
    h = L.conv(h, ksize=3, stride=1, f_in=layer_sizes[0], f_out=layer_sizes[1], seed=rng.randint(123456), name='c4')
    h = L.lrelu(bn(h, layer_sizes[1], is_training=is_training, update_batch_stats=update_batch_stats, name='b4'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=3, stride=1, f_in=layer_sizes[1], f_out=layer_sizes[1], seed=rng.randint(123456), name='c5')
    h = L.lrelu(bn(h, layer_sizes[1], is_training=is_training, update_batch_stats=update_batch_stats, name='b5'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=3, stride=1, f_in=layer_sizes[1], f_out=layer_sizes[1], seed=rng.randint(123456), name='c6')
    h = L.lrelu(bn(h, layer_sizes[1], is_training=is_training, update_batch_stats=update_batch_stats, name='b6'), FLAGS.lrelu_a)

    h = L.max_pool(h, ksize=2, stride=2)
    
    h = tf.nn.dropout(h, keep_prob=0.5, seed=rng.randint(123456)) if stochastic else h
    
    h = L.conv(h, ksize=3, stride=1, f_in=layer_sizes[1], f_out=layer_sizes[2], seed=rng.randint(123456), padding="VALID", name='c7')
    h = L.lrelu(bn(h, layer_sizes[2], is_training=is_training, update_batch_stats=update_batch_stats, name='b7'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=1, stride=1, f_in=layer_sizes[2], f_out=layer_sizes[3], seed=rng.randint(123456), name='c8')
    h = L.lrelu(bn(h, layer_sizes[3], is_training=is_training, update_batch_stats=update_batch_stats, name='b8'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=1, stride=1, f_in=layer_sizes[3], f_out=layer_sizes[4], seed=rng.randint(123456), name='c9')
    h = L.lrelu(bn(h, layer_sizes[4], is_training=is_training, update_batch_stats=update_batch_stats, name='b9'), FLAGS.lrelu_a)

    h = tf.reduce_mean(h, reduction_indices=[1, 2])  # Global average pooling

    return h

def forward_after_adt(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):
    h = x
    rng = numpy.random.RandomState(seed)
    
    h = L.fc(h, layer_sizes[4], 10, seed=rng.randint(123456), name='fc')

    if FLAGS.top_bn:
        h = bn(h, 10, is_training=is_training,
                 update_batch_stats=update_batch_stats, name='bfc')

    return h


def adversarial_dropout(x, cur_mask, Jacobian, delta, name="ad"):
    
    dim = tf.reduce_prod(tf.shape(x)[1:])
    change_limit = int(layer_sizes[4]*delta)
    changed_mask = cur_mask
    
    if change_limit != 0 :
        
        dir = tf.reshape(Jacobian, [-1, dim])
        
        # mask (cur_mask=1->m=1), (cur_mask=0->m=-1)
        m = cur_mask
        m = 2.0*m - tf.ones_like(m)
        
        # sign of Jacobian  (J>0 -> s=1), (J<0 -> s= -1)
        s = tf.cast( tf.greater(dir, float(0.0)), tf.float32)
        s = 2.0*s - tf.ones_like(s)                    
        
        # remain (J>0, m=-1) and (J<0, m=1), which are candidates to be changed
        change_candidate = tf.cast( tf.less( s*m, float(0.0) ), tf.float32) # s = -1, m = 1
        ads_dL_dx = tf.abs(dir)
        
        # ordering abs_Jacobian for candidates
        # the maximum number of the changes is "change_limit"
        # draw top_k elements ( if the top k element is 0, the number of the changes is less than "change_limit" ) 
        left_values = change_candidate*ads_dL_dx
        with tf.device("/cpu:0"):
            min_left_values = tf.nn.top_k(left_values, change_limit+1)[0][:,-1]    
        change_target = tf.cast(  tf.greater(left_values, tf.expand_dims(min_left_values, -1) ), tf.float32)
        
        # changed mask with change_target
        changed_mask = (m - 2.0*m*change_target + tf.ones_like(m))*0.5 
        
        # normalization
        sum_mask = tf.cast( tf.reduce_sum(changed_mask, axis=1, keep_dims=True), tf.float32)
        changed_mask = math_ops.div(changed_mask,sum_mask)*tf.to_float(dim)
        changed_mask = tf.reshape(changed_mask, tf.shape(x))
    
    ret = x * changed_mask
    
    return ret, changed_mask


def logit_adversarial_dropout(x, y, ehta, is_training=True, update_batch_stats=True, _loss_fuc_=L.kl_divergence_with_logit,  seed=1234):
    
    h = x
    h = forward_before_adt(h, is_training=is_training, update_batch_stats=update_batch_stats, stochastic=True, seed=seed)
    temp_y = forward_after_adt(h, is_training=is_training, update_batch_stats=False, stochastic=True, seed=seed)
    
    # second adv
    temp_loss = _loss_fuc_(temp_y, y)
    
    dL_dh = tf.gradients(temp_loss, [h], aggregation_method=2)[0]
    dL_dh = tf.stop_gradient(dL_dh)
    Jacobian = h*dL_dh
    cur_mask = tf.ones_like(h)
    
    h, second_adv_mask = adversarial_dropout(h, cur_mask, Jacobian, ehta, name="ad3")
    h = forward_after_adt(h, is_training=is_training, update_batch_stats=update_batch_stats, stochastic=True, seed=seed)

    return h 
     

if __name__ == "__main__":
    tf.app.run()
