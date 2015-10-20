#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from model import Model

sigmoid = T.nnet.sigmoid


class SRNN(Model):

    def __init__(self, name, numvis, numhid, numframes, output_type='real', cheating_level=.0, numpy_rng=None, theano_rng=None):
        super(SRNN, self).__init__(name=name)

        # store arguments
        self.numvis = numvis
        self.numhid = numhid
        self.numframes = numframes
        self.output_type = output_type
        self.cheating_level = theano.shared(np.float32(cheating_level))

        if not numpy_rng:
            self.numpy_rng = np.random.RandomState(1)
        else:
            self.numpy_rng = numpy_rng
        if not theano_rng:
            self.theano_rng = RandomStreams(1)
        else:
            self.theano_rng = theano_rng

        # create input var
        self.inputs = T.matrix(name='inputs')

        # set up params
        self.whh = theano.shared(value=np.eye(self.numhid).astype(theano.config.floatX), name='whh')
        self.whx = theano.shared(value=self.numpy_rng.uniform( low=-0.01, high=0.01, size=(self.numhid, numvis)).astype(theano.config.floatX), name='whx')
        self.wxh = theano.shared(value=self.numpy_rng.uniform( low=-0.01, high=0.01, size=(numvis, self.numhid)).astype(theano.config.floatX), name='wxh')
        self.bx = theano.shared(value=0.0 * np.ones( numvis, dtype=theano.config.floatX), name='bx')
        self.bhid = theano.shared(value=0.0 * np.ones( self.numhid, dtype=theano.config.floatX), name='bhid')
        self.params = [self.whh, self.whx, self.wxh, self.bx, self.bhid]

        self._batchsize = self.inputs.shape[0]
        # reshape to TxBxN (time, batch, numvis)
        self._input_frames = self.inputs.reshape(( self._batchsize, self.inputs.shape[1] // self.numvis, self.numvis)).transpose(1, 0, 2)

        #one-step prediction for sampling-
        self.hids_0 = T.zeros((self._batchsize, self.numhid))
        self.hids_1 = T.dot(self.hids_0, self.whh) + self.bhid + T.dot(self._input_frames[0], self.wxh)
        self.hids_1 = self.hids_1 * (self.hids_1 > 0)
        self.x_pred_1 = T.dot(self.hids_1, self.whx) + self.bx
        #-one-step prediction for sampling 

        def step(x_gt_t, x_tm1, hids_tm1):
            pre_hids_t = T.dot(hids_tm1, self.whh) + self.bhid + T.dot(self.cheating_level*x_gt_t+(1.-self.cheating_level)*x_tm1, self.wxh)
            hids_t = pre_hids_t * (pre_hids_t > 0)
            x_pred_t = T.dot(hids_t, self.whx) + self.bx
            return x_pred_t, hids_t

        (self._predictions, self.hids), self.updates = theano.scan(
                                                        fn=step,
                                                        sequences=self._input_frames,
                                                        outputs_info=[self._input_frames[0], self.hids_0])

        if self.output_type == 'real':
            self._prediction = self._predictions[:, :, :self.numvis]
        elif self.output_type == 'binary':
            self._prediction = sigmoid(self._predictions[:, :, :self.numvis])
        elif self.output_type == 'softmax':
            # softmax doesn't support 3d tensors, reshape batch and time axis
            # together, apply softmax and reshape back to 3d tensor
            self._prediction = T.nnet.softmax(
                self._predictions[:, :, :self.numvis].reshape((
                    self._predictions.shape[0] * self._predictions.shape[1],
                    self.numvis))).reshape((
                        self._predictions.shape[0], self._predictions.shape[1],
                        self.numvis))
        else:
            raise ValueError('unsupported output_type')

        self._prediction_for_training = self._prediction[:self.numframes-1]

        if self.output_type == 'real':
            self._cost = T.mean((
                self._prediction_for_training -
                self._input_frames[1:self.numframes]
            )**2)
            self._cost_varlen = T.mean((
                self._prediction -
                self._input_frames[1:]
            )**2)
        elif self.output_type == 'binary':
            self._cost = -T.mean(
                self._input_frames[1:self.numframes] *
                T.log(self._prediction_for_training) +
                (1.0 - self._input_frames[1:self.numframes]) * T.log(
                    1.0 - self._prediction))
            self._cost_varlen = -T.mean(
                self._input_frames[1:] *
                T.log(self._prediction_for_training) +
                (1.0 - self._input_frames[1:]) * T.log(
                    1.0 - self._prediction))
        elif self.output_type == 'softmax':
            self._cost = -T.mean(T.log(
                self._prediction_for_training) *
                self._input_frames[1:self.numframes])
            self._cost_varlen = -T.mean(T.log(
                self._prediction) *
                self._input_frames[1:])

        self._grads = T.grad(self._cost, self.params)

        self.inputs_var = T.fmatrix('inputs_var')
        self.nsteps = T.lscalar('nsteps')
        givens = {}
        givens[self.inputs] = T.concatenate((
            self.inputs_var[:, :self.numvis], T.zeros((
                self.inputs_var.shape[0],
                self.nsteps*self.numvis))), axis=1)

        self.predict = theano.function(
            [self.inputs_var, theano.Param(
                self.nsteps, default=self.numframes-4)],
            self._prediction.transpose(1, 0, 2).reshape((
                self.inputs_var.shape[0], self.nsteps*self.numvis)),
            updates=self.updates, givens=givens)
        self.cost = theano.function(
            [self.inputs], self._cost,
            updates=self.updates)
        self.grads = theano.function(
            [self.inputs], self._grads,
            updates=self.updates)

    def grad(self, x):
        def get_cudandarray_value(x):
            if type(x) == theano.sandbox.cuda.CudaNdarray:
                return np.array(x.__array__()).flatten()
            else:
                return x.flatten()
        return np.concatenate([get_cudandarray_value(g) for g in self.grads(x)])

    def sample(self, numcases=1, numframes=10, temperature=1.0):
        assert self.output_type == 'softmax'
        #first_prediction_and_state = theano.function([], [self.theano_rng.multinomial(pvals=T.nnet.softmax(self.x_pred_4/temperature)), self.hids_1])
        next_prediction_and_state = theano.function([self._input_frames, self.hids_0], [self.theano_rng.multinomial(pvals=T.nnet.softmax(self.x_pred_1/temperature)), self.hids_1])

        preds = np.zeros((numcases, numframes, self.numvis), dtype="float32")
        preds[:,0,:] = self.numpy_rng.multinomial(numcases, pvals=np.ones(self.numvis)/np.float(self.numvis))
        hids = np.zeros((numcases, self.numhid), dtype="float32")
        #nextpredandstate = next_prediction_and_state(frames, np.zeros((numcases, self.numhid), dtype="float32"))
        #frames[0][:,:] = nextpredandstate[0][:,:self.numvis]
        #preds[:,0*self.numvis:1*self.numvis] = nextpredandstate[0][:,:self.numvis]
        for t in range(1, numframes):
            nextpredandstate = next_prediction_and_state(preds[:,[t-1],:], hids)
            hids = nextpredandstate[1]
            preds[:,t,:] = nextpredandstate[0]
        return preds



