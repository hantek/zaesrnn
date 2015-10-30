#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from model import Model

sigmoid = T.nnet.sigmoid


class SRNN(Model):

    def __init__(self, name, numvis, numhid, numlayers, numframes, output_type='real', dropout=0.0, numpy_rng=None, theano_rng=None):
        super(SRNN, self).__init__(name=name)

        self.numvis = numvis            # frame length * alphabet size (1 * 27)
        self.numhid = numhid            # 512
        self.numlayers = numlayers      # 3
        self.numframes = numframes      # maxnumframes (100)
        self.output_type = output_type  # softmax
        self.dropout = dropout          # 0.5

        if not numpy_rng:
            self.numpy_rng = np.random.RandomState(1)
        else:
            self.numpy_rng = numpy_rng
        if not theano_rng:
            self.theano_rng = RandomStreams(1)
        else:
            self.theano_rng = theano_rng

        self.inputs = T.matrix(name='inputs')

        self.whh = [theano.shared(value=np.eye(self.numhid).astype(theano.config.floatX), name='whh'+str(k)) for k in range(self.numlayers)]
        self.whx = [theano.shared(value=self.numpy_rng.uniform( low=-0.01, high=0.01, size=(self.numhid, self.numvis)).astype(theano.config.floatX), name='whx'+str(k)) for k in range(self.numlayers)]
        self.wxh = [theano.shared(value=self.numpy_rng.uniform( low=-0.01, high=0.01, size=(self.numvis, self.numhid)).astype(theano.config.floatX), name='wxh'+str(0))]
        self.wxh = self.wxh + [theano.shared(value=self.numpy_rng.uniform( low=-0.01, high=0.01, size=(self.numhid, self.numhid)).astype(theano.config.floatX), name='wxh'+str(k)) for k in range(self.numlayers-1)]
        self.bx = theano.shared(value=0.0 * np.ones( self.numvis, dtype=theano.config.floatX), name='bx')
        self.bhid = [theano.shared(value=0.0 * np.ones( self.numhid, dtype=theano.config.floatX), name='bhid'+str(k)) for k in range(self.numlayers)]
        self.params = self.whh + self.whx + self.wxh + self.bhid + [self.bx]

        self._batchsize = self.inputs.shape[0]
        self._input_frames = self.inputs.reshape(( self._batchsize, self.inputs.shape[1] // self.numvis, self.numvis)).transpose(1, 0, 2)

        #1-step prediction --- 
        self.hids_0 = T.zeros((self._batchsize, self.numhid*self.numlayers)) 
        self.hids_1 = [T.dot(self.hids_0[:,:self.numhid], self.whh[0]) + self.bhid[0] + T.dot(self._input_frames[0], self.wxh[0])]
        self.hids_1[0] *= (self.hids_1[0] > 0)
        for k in range(1, self.numlayers):
            self.hids_1.append(T.dot(self.hids_0[:,k*self.numhid:(k+1)*self.numhid], self.whh[k]) + self.bhid[k] + T.dot(self.hids_1[k-1], self.wxh[k]))
            self.hids_1[-1] *= (self.hids_1[-1] > 0)

        self.x_pred_1 = self.bx 
        for k in range(self.numlayers):
            self.x_pred_1 += T.dot(self.hids_1[k], self.whx[k]) 
        self.hids_1 = T.concatenate(self.hids_1, 1)
        #--- 1-step prediction 

        def step_dropout(x_gt_t, dropoutmask, x_tm1, hids_tm1):
            hids_tm1 = [hids_tm1[:,k*self.numhid:(k+1)*self.numhid] for k in range(self.numlayers)]
            pre_hids_t = [T.dot(hids_tm1[0], self.whh[0]) + self.bhid[0] + T.dot(x_gt_t, self.wxh[0])]
            hids_t = [pre_hids_t[0] * (pre_hids_t[0] > 0)]
            for k in range(1, self.numlayers):
                pre_hids_t.append(T.dot(hids_tm1[k], self.whh[k]) + self.bhid[k] + T.dot(dropoutmask*hids_t[k-1], (1.0/self.dropout)*self.wxh[k]))
                hids_t.append(pre_hids_t[k] * (pre_hids_t[k] > 0))
            x_pred_t = self.bx
            for k in range(self.numlayers):
                x_pred_t += T.dot(hids_t[k], self.whx[k]) 
            return x_pred_t, T.concatenate(hids_t, 1)

        def step_nodropout(x_gt_t, x_tm1, hids_tm1):
            hids_tm1 = [hids_tm1[:,k*self.numhid:(k+1)*self.numhid] for k in range(self.numlayers)]
            pre_hids_t = [T.dot(hids_tm1[0], self.whh[0]) + self.bhid[0] + T.dot(x_gt_t, self.wxh[0])]
            hids_t = [pre_hids_t[0] * (pre_hids_t[0] > 0)]
            for k in range(1, self.numlayers):
                pre_hids_t.append(T.dot(hids_tm1[k], self.whh[k]) + self.bhid[k] + T.dot(hids_t[k-1], self.wxh[k]))
                hids_t.append(pre_hids_t[k] * (pre_hids_t[k] > 0))
            x_pred_t = self.bx
            for k in range(self.numlayers):
                x_pred_t += T.dot(hids_t[k], self.whx[k]) 
            return x_pred_t, T.concatenate(hids_t, 1)

        if self.dropout == 0.0:
            (self._predictions, self.hids), self.updates = theano.scan(
                                                        fn=step_nodropout,
                                                        sequences=self._input_frames,
                                                        outputs_info=[self._input_frames[0], self.hids_0])
        else:
            self._dropoutmask = theano_rng.binomial(
                size=(self.inputs.shape[1] // self.numvis,
                      self._batchsize, self.numhid),
                n=1, p=self.dropout, dtype=theano.config.floatX
            )
            (self._predictions, self.hids), self.updates = theano.scan(
                                                        fn=step_dropout,
                                                        sequences=[self._input_frames, self._dropoutmask],
                                                        outputs_info=[self._input_frames[0], self.hids_0])

        if self.output_type == 'real':
            self._prediction = self._predictions[:, :, :self.numvis]  # dims: [time step, batch idx, numvis]
        elif self.output_type == 'binary':
            self._prediction = sigmoid(self._predictions[:, :, :self.numvis])
        elif self.output_type == 'softmax':
            # softmax doesn't support 3d tensors, reshape batch and time axis
            # together, apply softmax and reshape back to 3d tensor
            self._prediction = T.nnet.softmax(
                self._predictions[:, :, :self.numvis].reshape((
                    self._predictions.shape[0] * self._predictions.shape[1],
                    self.numvis
                ))
            ).reshape((
                self._predictions.shape[0],
                self._predictions.shape[1],
                self.numvis
            ))
        else:
            raise ValueError('unsupported output_type')

        self._prediction_for_training = self._prediction[:self.numframes-1]

        if self.output_type == 'real':
            self._cost = T.mean(( self._prediction_for_training - self._input_frames[1:self.numframes])**2)
            self._cost_varlen = T.mean(( self._prediction - self._input_frames[1:])**2)  # for various lengths
        elif self.output_type == 'binary':
            self._cost = -T.mean( self._input_frames[1:self.numframes] * T.log(self._prediction_for_training) + (1.0 - self._input_frames[1:self.numframes]) * T.log( 1.0 - self._prediction))
            self._cost_varlen = -T.mean( self._input_frames[1:] * T.log(self._prediction_for_training) + (1.0 - self._input_frames[1:]) * T.log( 1.0 - self._prediction))
        elif self.output_type == 'softmax':
            self._cost = -T.mean(T.log( self._prediction_for_training) * self._input_frames[1:self.numframes])
            self._cost_varlen = -T.mean(T.log( self._prediction) * self._input_frames[1:])

        self._grads = T.grad(self._cost, self.params)

        self.inputs_var = T.fmatrix('inputs_var')
        self.nsteps = T.lscalar('nsteps')
        givens = {}
        givens[self.inputs] = T.concatenate(
            ( self.inputs_var[:, :self.numvis],
              T.zeros((self.inputs_var.shape[0], self.nsteps*self.numvis))
            ),
            axis=1)
        
        # predict given the first letters. 
        self.predict = theano.function(
            [self.inputs_var, theano.Param(self.nsteps, default=self.numframes-4)],
            self._prediction.transpose(1, 0, 2).reshape((self.inputs_var.shape[0], self.nsteps*self.numvis)),
            updates=self.updates, givens=givens)
        self.cost = theano.function( [self.inputs], self._cost, updates=self.updates)
        self.grads = theano.function( [self.inputs], self._grads, updates=self.updates)

    def grad(self, x):
        def get_cudandarray_value(x):
            if type(x) == theano.sandbox.cuda.CudaNdarray:
                return np.array(x.__array__()).flatten()
            else:
                return x.flatten()
        return np.concatenate([get_cudandarray_value(g) for g in self.grads(x)])

    def sample(self, numcases=1, numframes=10, temperature=1.0):
        assert self.output_type == 'softmax'
        next_prediction_and_state = theano.function([self._input_frames, self.hids_0], [self.theano_rng.multinomial(pvals=T.nnet.softmax(self.x_pred_1/temperature)), self.hids_1])
        preds = np.zeros((numcases, numframes, self.numvis), dtype="float32")
        preds[:,0,:] = self.numpy_rng.multinomial(numcases, pvals=np.ones(self.numvis)/np.float(self.numvis))
        hids = np.zeros((numcases, self.numhid*self.numlayers), dtype="float32")
        for t in range(1, numframes):
            nextpredandstate = next_prediction_and_state(preds[:,[t-1],:], hids)
            hids = nextpredandstate[1]
            preds[:,t,:] = nextpredandstate[0]
        return preds



