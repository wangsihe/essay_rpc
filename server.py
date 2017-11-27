import tensorflow as tf
import re
import tensorlayer as tl
from tensorlayer.layers import *
from scipy.misc import imread, imresize
import cv2
import numpy as np
import json
from sklearn.externals import joblib
import zerorpc

from model.tfl import conv_layers_simple_api, fc_layers 
from func.patch_from_image import patch_from_image

def softmax(x):
    return (np.exp(x).transpose() / np.sum(np.exp(x), 1)).transpose()

def clip(x, maxv):
    x = np.array(x)
    x[x > maxv] = maxv
    return np.int32(np.round(x))

class IM2RES(object):
    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 1], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

    net_in = InputLayer(x, name='input')
    net_cnn = conv_layers_simple_api(net_in)  # simplified CNN APIs
    network = fc_layers(net_cnn)
    y = network.outputs

    cost = tl.cost.cross_entropy(y, y_, 'cost')
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    load_params = tl.files.load_npz(name='./model/300_75_bw/hw_189.npz')
    tl.files.assign_params(sess, load_params, network)

    clf = joblib.load('./clfmodel/junior.m')

    def im2res(self, im, text, yigai, maxv):
        nparr = np.fromstring(im, np.int8)
        im = cv2.imdecode(nparr, 0)

        words_count = len([w for w in re.split('[\s,\.]+', text) if w.isalpha()])
        if words_count < 10:
            return 0
        imts = patch_from_image(im, 300, 150)
        batch_size = 1
        p = []

        if len(imts) == 0:
            c = np.zeros(20, np.float32)
        else:
            tmpts = np.zeros(len(imts), np.int64)
            for patch, tmpy in tl.iterate.minibatches(imts, tmpts, batch_size, shuffle=False):
                prob, lb = self.sess.run([self.y, self.y_], feed_dict={self.x: patch, self.y_: tmpy})
                prob[prob > 100] = 100
                prob[prob < -100] = -100
                prob = softmax(prob)
                p.append(prob)
            if len(p) == 0:
                c = np.zeros(20, np.float32)
            else:
                p = np.reshape(np.squeeze(p), [-1, 5])

                pres = np.argmax(p, 1)
                hist = np.zeros(5, np.float32)
                for i in pres:
                    hist[np.int32(i)] += 1
                probsum = np.sum(p, 0)
                c1 = np.concatenate((hist, probsum))    

                p = p[:10, :]
                pres = np.argmax(p, 1)
                hist = np.zeros(5, np.float32)
                for i in pres:
                    hist[np.int32(i)] += 1
                probsum = np.sum(p, 0)
                c2 = np.concatenate((hist, probsum))    
                c = np.concatenate((c1, c2))
        wn = 100
        if not yigai.has_key('feedbacksLite'):
            yigai['feedbacksLite'] = list()

        ft = [
            yigai['developmentScore'],
            yigai['grammarScore'],
            yigai['lexicalComplexityScore'],
            yigai['lexicalRelevanceScore'],
            yigai['mechanicsScore'],
            yigai['organizationScore'],
            yigai['styleScore'],
            yigai['usageScore'],
            1.0 * len(yigai['checkerResults']) / max(1, words_count),
            1.0 * len(yigai['lexicalEnrichments']) / max(1, words_count),
            1.0 * words_count / wn,#10
            ([f['score'] for f in yigai['feedbacksLite'] if f['section'] == 'SYNTAX'] + [0])[0],
            ([f['score'] for f in yigai['feedbacksLite'] if f['section'] == 'CONNECTIVES'] + [0])[0],
            ([f['score'] for f in yigai['feedbacksLite'] if f['section'] == 'CONTENT'] + [0])[0],
        ]

        if sum(c) == 0:
            c = np.zeros(20)
        else:
            c[0:5] = c[0:5] / np.max(c[0:5])
            c[5:10] = c[5:10] / np.max(c[5:10])
            c[10:15] = c[10:15] / np.max(c[10:15])
            c[15:20] = c[15:20] / np.max(c[15:20])

        ft.extend(list(c))
        s = self.clf.predict([ft])
        res = clip(s * maxv, maxv)
        return str(res[0])

s = zerorpc.Server(IM2RES())
s.bind('tcp://192.168.1.115:1134')
s.run()

