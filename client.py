import tensorflow as tf
import re
import tensorlayer as tl
from tensorlayer.layers import *
from scipy.misc import imread, imresize
import cv2
import numpy as np
import json
from sklearn.externals import joblib

from model.tfl import conv_layers_simple_api, fc_layers 
from func.patch_from_image import patch_from_image

import zerorpc

c = zerorpc.Client()
c.connect('tcp://192.168.1.115:1134')

dataf = './'
ex = '4b818ef131'
jsonf = './json/' + ex + '.json'
maxv = 25
text = json.loads(open(dataf + jsonf).read())

res = []
mes = []
imdir = []
for essay in text['data']:
    s = dataf + 'essay_data_ts/' + ex
    fname = essay['uid'].encode('utf-8')
    try:
        yigai = json.loads(open('%s/%s.nlp' % (s, fname)).read())
    except:
        continue

    text_ = essay['text'].encode('utf-8')
    im = cv2.imread('%s/%s.png' % (s, fname), 0)
    im_encode = cv2.imencode('.png', im)[1].tostring()

    res.append(np.int32(c.im2res(im_encode, text_, yigai, maxv)))

def myhist(ms, maxv):
    cm = np.zeros((maxv + 1), np.int32)
    for i in ms:
        cm[i] += 1
    return cm

print (list(myhist(res, maxv)), len(text['data']))

