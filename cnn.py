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

def logit(x, dropout_mask=None, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):

    rng = numpy.random.RandomState(seed)
    
    h = L.gl(x, std=FLAGS.sigma)
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

    # dropout with mask
    if dropout_mask is not None:
        # When we hold a dropout mask (Fully Connected)
        h = h*dropout_mask
    else:
        # Base dropout mask is 1 (Fully Connected)
        dropout_mask = tf.ones_like(h) 

    h = L.fc(h, layer_sizes[4], 10, seed=rng.randint(123456), name='fc')

    if FLAGS.top_bn:
        h = bn(h, 10, is_training=is_training,
                 update_batch_stats=update_batch_stats, name='bfc')

    return h, dropout_mask
     

if __name__ == "__main__":
    tf.app.run()
