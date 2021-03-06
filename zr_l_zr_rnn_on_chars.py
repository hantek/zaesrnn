import os
import sys 
#sys.path.append("..")
sys.path.append("../downup/")
import zr_l_zr_rnn
import graddescent_rewrite

import pylab
import numpy
import numpy.random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import pdb


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


def vec2chars(vec, invdictionary):
    return [invdictionary[c] for c in unhot(vec.reshape(-1,27))]


print 'loading data...'
framelen = 1
maxnumframes = 50
alphabetsize = 27
text = open("/data/lisatmp3/zlin/trunk/text8").readline()

allletters = 'abcdefghijklmnopqrstuvwxyz '
text = ''.join([a+b for b in allletters for a in allletters]) + text  

dictionary = dict(zip(list(set(text)), range(alphabetsize)))
invdict = {v: k for k, v in dictionary.items()}
numtrain, numvalid = maxnumframes*400000, maxnumframes*10000
#numtrain, numvalid = 10000000, 200000
train_features_numpy = onehot(numpy.array([dictionary[c] for c in text[:numtrain]])).reshape(-1, 27*maxnumframes*framelen)
valid_features_numpy = onehot(numpy.array([dictionary[c] for c in text[numtrain:numtrain+numvalid]])).reshape(-1, 27*maxnumframes*framelen)
numcases = train_features_numpy.shape[0]
del text 
print 'Done.'

numpy_rng.shuffle(train_features_numpy)
numpy_rng.shuffle(valid_features_numpy)
#train_features = theano.shared(train_features_numpy, name='train_set', borrow=True)
valid_features = theano.shared(valid_features_numpy, name='valid_set', borrow=True)

model = zr_l_zr_rnn.SRNN(name="aoeu",
                         numvis=framelen*alphabetsize,
                         numsz=1024,
                         numrz=None,
                         numsl=256,
                         numrl=None,
                         numframes=50,
                         cheating_level=0., 
                         output_type="softmax",
                         numpy_rng=numpy_rng, 
                         theano_rng=theano_rng)

ppw = 2 ** T.mean(  # first mean NLL over each time step prediction, then mean over the whole batch
    -T.log2(  # apply log_2
        T.sum(  # summing over the 3rd dimention, which has 27 elements
            (model._prediction_for_training * model._input_frames[1:]),
            axis=2
        )
    )
)
#train_perplexity = theano.function([], ppw, givens={model.inputs:train_features})
valid_perplexity = theano.function([], ppw, givens={model.inputs:valid_features})

#model.monitor = model.normalizefilters


# TRAIN MODEL
trainer = graddescent_rewrite.SGD_Trainer(model,
                                          train_features_numpy,
                                          batchsize=50,
                                          learningrate=0.1,
                                          loadsize=50000,
                                          gradient_clip_threshold=1.0)

f=open('zr_l_zr_rnn_on_chars_perplexity.log', 'w')
crnt_ppw = valid_perplexity()
print "BEFORE_TRAINING: valid perplexity: %f" % (crnt_ppw)
f.write(str(crnt_ppw)+'\n')
print 'training...'
for epoch in xrange(100):
    epccost = trainer.step()
    save_params(model, 'zr_l_zr_rnn_on_chars_params.npy')
    # print "perplexity train: %f, valid: %f" % (train_perplexity(), valid_perplexity())
    crnt_ppw = valid_perplexity()
    print "valid perplexity: %f" % (crnt_ppw)
    f.write(str(crnt_ppw)+'\n')

f.close()
print "sampling from the model:" + ''.join(vec2chars(model.sample(numframes=1000), invdict))
pdb.set_trace()
