from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from scipy.io import loadmat

import numpy as np
from scipy import linalg
import glob
import pickle

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib

import tensorflow as tf
from dataset_utils import *

DATA_URL_TRAIN = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
DATA_URL_TEST = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'dataset/svhn', "")
tf.app.flags.DEFINE_integer('num_labeled_examples', 73257, "The number of labeled examples")
tf.app.flags.DEFINE_integer('dataset_seed', 1, "dataset seed")

NUM_EXAMPLES_TRAIN = 73257
NUM_EXAMPLES_TEST = 26032


def maybe_download_and_extract():
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    filepath_train_mat = os.path.join(FLAGS.data_dir, 'train_32x32.mat')
    filepath_test_mat = os.path.join(FLAGS.data_dir, 'test_32x32.mat')
    if not os.path.exists(filepath_train_mat) or not os.path.exists(filepath_test_mat):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        urllib.request.urlretrieve(DATA_URL_TRAIN, filepath_train_mat, _progress)
        urllib.request.urlretrieve(DATA_URL_TEST, filepath_test_mat, _progress)

    # Training set
    print("Loading training data...")
    print("Preprocessing training data...")
    train_data = loadmat(FLAGS.data_dir + '/train_32x32.mat')
    train_x = (-127.5 + train_data['X']) / 255.
    train_x = train_x.transpose((3, 0, 1, 2))
    train_x = train_x.reshape([train_x.shape[0], -1])
    train_y = train_data['y'].flatten().astype(np.int32)
    train_y[train_y == 10] = 0

    # Test set
    print("Loading test data...")
    test_data = loadmat(FLAGS.data_dir + '/test_32x32.mat')
    test_x = (-127.5 + test_data['X']) / 255.
    test_x = test_x.transpose((3, 0, 1, 2))
    test_x = test_x.reshape((test_x.shape[0], -1))
    test_y = test_data['y'].flatten().astype(np.int32)
    test_y[test_y == 10] = 0

    np.save('{}/train_images'.format(FLAGS.data_dir), train_x)
    np.save('{}/train_labels'.format(FLAGS.data_dir), train_y)
    np.save('{}/test_images'.format(FLAGS.data_dir), test_x)
    np.save('{}/test_labels'.format(FLAGS.data_dir), test_y)


def load_svhn():
    maybe_download_and_extract()
    train_images = np.load('{}/train_images.npy'.format(FLAGS.data_dir)).astype(np.float32)
    train_labels = np.load('{}/train_labels.npy'.format(FLAGS.data_dir)).astype(np.float32)
    test_images = np.load('{}/test_images.npy'.format(FLAGS.data_dir)).astype(np.float32)
    test_labels = np.load('{}/test_labels.npy'.format(FLAGS.data_dir)).astype(np.float32)
    return (train_images, train_labels), (test_images, test_labels)


def prepare_dataset():
    (train_images, train_labels), (test_images, test_labels) = load_svhn()
    dirpath = os.path.join(FLAGS.data_dir, 'seed' + str(FLAGS.dataset_seed))
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    rng = np.random.RandomState(FLAGS.dataset_seed)
    rand_ix = rng.permutation(NUM_EXAMPLES_TRAIN)
    print(rand_ix)
    _train_images, _train_labels = train_images[rand_ix], train_labels[rand_ix]

    convert_images_and_labels(_train_images, _train_labels,
                              os.path.join(dirpath, 'svhn_sup_labeled_train.tfrecords'))
    convert_images_and_labels(test_images,
                              test_labels,
                              os.path.join(dirpath, 'svhn_sup_test.tfrecords'))


def one_transformed_input(batch_size=100,
           train=True, 
           shuffle=True, num_epochs=None):
    
    if train:
        filenames = ['svhn_sup_labeled_train.tfrecords']
        num_examples = FLAGS.num_labeled_examples
    else:
        filenames = ['svhn_sup_test.tfrecords']
        num_examples = NUM_EXAMPLES_TEST

    filenames = [os.path.join('seed' + str(FLAGS.dataset_seed), filename) for filename in filenames]
    filename_queue = generate_filename_queue(filenames, FLAGS.data_dir, num_epochs)
    image, label = read(filename_queue)
    image = transform(tf.cast(image, tf.float32)) if train else image
    return generate_batch([image, label], num_examples, batch_size, shuffle)

def two_transformed_input(batch_size=100,
           train=True, 
           shuffle=True, num_epochs=None):
    
    if train:
        filenames = ['svhn_sup_labeled_train.tfrecords']
        num_examples = FLAGS.num_labeled_examples
    else:
        filenames = ['svhn_sup_test.tfrecords']
        num_examples = NUM_EXAMPLES_TEST

    filenames = [os.path.join('seed' + str(FLAGS.dataset_seed), filename) for filename in filenames]
    filename_queue = generate_filename_queue(filenames, FLAGS.data_dir, num_epochs)
    image, label = read(filename_queue)
    
    image_1 = transform(tf.cast(image, tf.float32)) if train else image
    image_2 = transform(tf.cast(image, tf.float32)) if train else image
    
    return generate_batch([image_1, image_2, label], num_examples, batch_size, shuffle)



def main(argv):
    prepare_dataset()


if __name__ == "__main__":
    tf.app.run()
