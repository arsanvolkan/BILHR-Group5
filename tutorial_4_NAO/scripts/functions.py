#!/usr/bin/env python

#Group 5
# Laurie Dubois
# Bjoern Doeschl
# Gonzalo Olguin
# Dawen Zhou
# Volkan Arsan

import numpy as np
import random


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        outputs =   np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))
        return outputs

    def backward(self, dout):
        sigmoid = 1 / (1 + np.exp(-dout))
        return sigmoid * (1 - sigmoid)

class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        outputs = np.maximum(0,x)
        return outputs

    def backward(self, dout):
        dout[dout<0] = 0
        return dout

class LeakyReLU:
    def __init__(self, strength):
        self.strength = strength
        pass

    def forward(self, x):
        outputs = np.maximum(self.strength * x,x)
        return outputs

    def backward(self, dout):
        dx = np.ones_like(dout)
        dx[dout < 0] = self.strength
        return dx

class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        outputs = np.tanh(x)
        return outputs

    def backward(self, dout):
        tan = np.tanh(dout)
        dout = 1- tan**2
        return dout

class Softmax:
    def __init__(self):
        pass

    def forward(self,x):
        e = np.exp(np.subtract(x,np.max(x)))
        x = e/np.sum(e)
        return x

    def backward(self, dout):

        #tan = np.tanh(dout)
        #dout = 1- tan**2
        #return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        out = np.ones(10)
        out = out[...,None]
        return out


""" below are loss functions"""
class MeanSquaredError:
    """Class which computes the mean squared error"""
    def __init__(self):
        pass
    def forward(self, y_pred, g_truth):
        self.cache_g = g_truth
        mse = np.mean(np.square(np.subtract(g_truth, y_pred)))
        e_prime = 2*(y_pred-g_truth)/1
        return mse,e_prime
    def backward(self, y_pred):
        y = self.cache_g
        #mse =  -1 * (np.sum((y - y_pred)*y_pred))
        mse = 2 * np.subtract(y_pred, y)
        return mse

class CrossEntropyLoss:
    def __init__(self):
        pass
    def forward(self, y_pred, g_truth):
        self.cache = y_pred
        n_samples = g_truth.shape[0]
        logp = - np.log(y_pred[g_truth.argmax(axis=0)])
        e = np.sum(logp)/n_samples
        return e,0

    def backward(self, dout):
        return np.dot(-dout, 1/self.cache)
