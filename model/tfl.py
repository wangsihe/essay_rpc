import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def conv_layers_simple_api(net_in):
    with tf.name_scope('preprocess') as scope:
        mean = tf.constant([128], dtype=tf.float32, shape=[1, 1, 1, 1], name='img_mean')
        net_in.outputs = net_in.outputs - mean
    """ conv1 """
    network = Conv2d(net_in, n_filter=32, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool1')
    """ conv2 """
    network = Conv2d(network, n_filter=64, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool2')
    """ conv3 """
    network = Conv2d(network, n_filter=64, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool3')
    """ conv4 """
    network = Conv2d(network, n_filter=128, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool4')
    """ conv5 """
    network = Conv2d(network, n_filter=128, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool5')
    return network
def fc_layers(net):
    network = FlattenLayer(net, name='flatten')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc1_relu')
    network = DenseLayer(network, n_units=5, act=tf.identity, name='res')
    return network
