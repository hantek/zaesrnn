import os
HOME = os.environ['HOME']
import sys 
sys.path.append("../downup/")
import srnnmultilayer_scan
import graddescent_rewrite

import pdb

import pylab
import numpy
import numpy.random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)


def onehot(x,numclasses=None):                                                                                
    x = numpy.array(x)
    if x.shape==():                                                                                           
        x = x[numpy.newaxis]
    if numclasses is None:
        numclasses = x.max() + 1
    result = numpy.zeros(list(x.shape) + [numclasses],dtype=numpy.float32)
    z = numpy.zeros(x.shape)
    for c in range(numclasses):
        z *= 0
        z[numpy.where(x==c)] = 1
        result[...,c] += z
    return result                                                                                             


def unhot(labels):
    return labels.argmax(len(labels.shape)-1)                                                     


print '... loading data'
framelen = 1
maxnumframes = 100
#text = open(HOME+"/research/3rdparty/word2vec/trunk/text8").readline()
#text = open("/data/lisatmp3/zlin/trunk/text8").readline()
#traintext = open("/data/text/enwik8").read()[:96000000]
#traintext = open("/data/text/enwik8").read()[:10000000]
#testtext = open("/data/text/enwik8").read()[96000000:]
#traintext = open("./linux_input.txt").read()
traintext = open("/data/lisatmp3/zlin/trunk/text8").readline()

alphabet = list(set(traintext))
alphabetsize = len(alphabet)
dictionary = dict(zip(alphabet, range(alphabetsize)))
invdict = {v: k for k, v in dictionary.items()}
#numtrain, numvalid = maxnumframes*10000000, maxnumframes*10000
traintext = traintext[:len(traintext)-len(traintext)%(alphabetsize*maxnumframes*framelen)]
print "Done. Converting to one hot..."
train_features_numpy = onehot(numpy.array([dictionary[c] for c in traintext])).reshape(-1, alphabetsize*maxnumframes*framelen)
#valid_features_numpy = onehot(numpy.array([dictionary[c] for c in traintext[numtrain:numtrain+numvalid]])).reshape(-1, alphabetsize*maxnumframes*framelen)
numcases = train_features_numpy.shape[0]
del traintext 
print 'Done.'

numpy_rng.shuffle(train_features_numpy)
#numpy_rng.shuffle(valid_features_numpy)

print "... instantiating model"
model = srnnmultilayer_scan.SRNN(name="aoeu", 
                                 numvis=framelen*alphabetsize,
                                 numhid=512, 
                                 numlayers=3, 
                                 numframes=maxnumframes, 
                                 dropout=0.5,
                                 output_type="softmax",
                                 numpy_rng=numpy_rng, 
                                 theano_rng=theano_rng)
                                 #zaecost=0.1,)

# TRAIN MODEL
trainer = graddescent_rewrite.SGD_Trainer(model, train_features_numpy, batchsize=32, learningrate=0.01, loadsize=10000, gradient_clip_threshold=1.0)

print '... training'
for epoch in xrange(10):
    trainer.step()

print "... generating sequences:\n"

pdb.set_trace()
