#!/usr/bin/env python

#Group 5
# Laurie Dubois
# Bjoern Doeschl
# Gonzalo Olguin
# Dawen Zhou
# Volkan Arsan

import csv
import numpy as np
import random
from functions import *

def clamp(min_val, max_val, val):
    """Clamps the value to max min"""
    return min(max_val, max(min_val, val))

class SimpleLayer:
    """ This class descirbes a simple fully connected layer with in_number as input number and out_number as number of neurons.
    init_function describes the function with whic the weights are initialized, alpha is the learning rate and act_function is the activation function of this layer"""
    def __init__(self, in_number, out_number, init_function, alpha, act_function):
        self.in_number = in_number
        self.out_number = out_number

        #learning rate
        self.alpha = alpha

        # init the weights
        self.weights, self.bias  = self.init_weights(init_function)
        self.bias = self.bias[...,None]


        """Get an object of the activation function class"""
        if act_function == "Sigmoid":
            self.act_function = Sigmoid()
        elif act_function == "ReLU":
            self.act_function = ReLU()
        elif act_function == "Tanh":
            self.act_function = Tanh()
        elif act_function == "LReLU":
            self.act_function = LeakyReLU(0.1)
        elif act_function == "Softmax":
            self.act_function = Softmax()
        else:
            #default is Sigmoid
            print("Error no activation function type specified: Set default to Sigmoid.")
            self.act_function = Sigmoid()

    def forward(self, x):
        """Computes act(input*x + bias)"""
        self.cache = x
        output = np.dot(self.weights, x) + self.bias
        output = self.act_function.forward(output)
        self.out = output
        return  output


    def backward(self,  dout):
        """Computes the backward pass for this layer"""
        da = self.act_function.backward(self.out)
        dout = dout * da

        dw = np.dot(dout, np.transpose(self.cache))
        db = dout

        dout = np.dot(np.transpose(self.weights), dout)

        self.weights = self.weights - self.alpha * dw
        self.bias = self.bias - self.alpha * db
        return dout


    def init_weights(self, num_init):
        """Inits weights
            1 == random
            2 == all weights are 1
            3 == all weights are random numbers between 0.1 and 0
            4 == all weights are zero """
        if num_init == 1:
            return np.random.random((self.out_number,self.in_number)), np.random.random(self.out_number)
        elif num_init == 2:
            return np.ones((self.out_number,self.in_number)), np.ones(self.out_number)
        elif num_init == 3:
            return np.random.random((self.out_number,self.in_number))/10.0, np.random.random(self.out_number)/10.0
        else:
            return np.zeros((self.out_number,self.in_number)), np.zeros(self.out_number)

    def load_weights(self, weights, bias):
        """This function loads the trained weights, bias and overwrites its own local variables """
        self.weights = np.array(weights)
        self.bias = np.array(bias)

    def getWeightMatrix(self):
        return self.weights
    def getBiasVector(self):
        return self.bias


class Network:
    """This class defines the NN and holds the hidden layers."""
    def __init__(self, alpha):
        #init function says which method is used to init the weights
        # 1 == random | 2 == all weights are 1  | 3 == all weights are random between [0,0.1]else == all weighte are zero
        init_function = 1

        #learning rate
        self.alpha = alpha
        #initializes two layers, ine with 20 neurons and one with 2
        layer_1 = SimpleLayer(2, 20, init_function, self.alpha, "Sigmoid")
        layer_2 = SimpleLayer(20, 2, init_function, self.alpha, "Sigmoid")
        self.layers = []

        self.layers.append(layer_1)
        self.layers.append(layer_2)


    def forward(self, input):
        x = input
        #computes the forward pass of all layers but there is only one layer.
        i = 0
        for layer in self.layers:
            x = layer.forward(x)
            #print(i)
            i = i+1
        return x


    def backward(self, dout):
        #backward pass and adjust weights
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def save_weights(self):
        """This function stores the trained weights into an csv file.
            After training is done the weights can be loaded from that file instead of learn again.
            We have one csv file with weights trained on all 150 data points"""
        header=["weights"]
        name = './weights/weights.csv'

        with open(name, 'w') as outfile: #, newline=''
            writer=csv.writer(outfile)
            writer.writerow(header)
            for layer in self.layers:
                weight_store = layer.getWeightMatrix()
                bias = layer.getBiasVector()
                for row in weight_store:
                    writer.writerow(row)
                writer.writerow(["next"])
                for row in bias:
                    writer.writerow(row)
                writer.writerow(["next_b"])
            writer.writerow(["end"])
        print("done writing file, all weights have been written")

    def load_weights(self, weights, bias):
        """This function loads the weights into the layers"""
        i = 0
        for layer in self.layers:
            layer.load_weights(weights[i], bias[i])
            i = i+1
