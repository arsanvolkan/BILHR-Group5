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

def clamp(min_val, max_val, val):
    """Clamps the value to max min"""
    return min(max_val, max(min_val, val))

class SimpleLayer:
    """This class is the layer 3 in the CMAC it stores the weights and computes the prediction on the active neurons.
    Moreover this class also adjusts the weights."""
    def __init__(self, number_neurons, init_function, alpha, mode, receptive_field):
        self.number_neurons = number_neurons

        #learning rate, mode and receptive_field for backward computation
        self.alpha = alpha
        self.mode = mode
        self.receptive_field = receptive_field

        # init the weights
        self.weights_p = self.init_weights(init_function)
        self.weights_r = self.init_weights(init_function)


    def forward(self, id_neurons):
        """Computes weighted sum of the active neurons, norms it, scales it to the values of the joint_limits and returns"""
        # mult weights and sum
        sum_p = 0
        sum_r = 0

        #computes the weighted sum of all active neurons
        for id in id_neurons:
            sum_p = sum_p + self.weights_p[id-1]
            sum_r = sum_r + self.weights_r[id-1]

        angle_shoulder_pitch =  sum_p
        angle_shoulder_roll = sum_r
        return angle_shoulder_pitch,angle_shoulder_roll

    def backward(self, loss, miss_val, id_neurons):
        """adjust weights"""
        """loss is output of error function
            and miss_val is (ground truth - actual)"""

        for id in id_neurons:
            if self.mode == "error_training":
                
                self.weights_p[id-1] = self.weights_p[id-1] - (self.alpha/self.receptive_field * loss[0])
                self.weights_r[id-1] = self.weights_r[id-1] - (self.alpha/self.receptive_field * loss[1])
               
            elif self.mode == "target_training":
                self.weights_p[id-1] = self.weights_p[id-1] + (self.alpha/self.receptive_field * miss_val[0])
                self.weights_r[id-1] = self.weights_r[id-1] + (self.alpha/self.receptive_field * miss_val[1])
            else:
                print("Error in backwards function")


    def init_weights(self, num_init):
        """Inits weights
            1 == random
            2 == all weights are 0.25
            3 == all weights are zero """
        if num_init == 1:
            return np.random.random(self.number_neurons)
        elif num_init == 2:
            return np.ones(self.number_neurons)/13
        else:
            return np.zeros(self.number_neurons)

    def load_weights(self, weights_p, weights_r):
        """This function somewhat loads the trained weights and overwrites its variable """
        self.weights_p = weights_p
        self.weights_r = weights_r


    def save_weights(self, use_split):
        """This function stores the trained weights into an csv file.
            After training is done the weights can be loaded from that file instead of learn again.
            We have one csv file with weights trained on all 150 data points and one file with weights
            trained on 75 data points (use_split)"""
        header=["weights"]
        if use_split:
            name = './weights/weights_split.csv'
        else:
            name = './weights/weights.csv'

        with open(name, 'w')as outfile:#, newline='') as outfile:
            writer=csv.writer(outfile)
            writer.writerow(header)
            writer.writerow(self.weights_p)
            writer.writerow(self.weights_r)
        print("done writing file, all {} weights have been written".format(len(self.weights_p)))

class MeanSquaredError:
    """Class which computes the mean squared error"""
    def __init__(self):
        pass
    def forward(self, input, g_truth):
        #we need two erros, to return , retunrn a np array here, first for x1, second for x2
        mse=np.zeros(len(input))
        for i in range(len(input)):
            mse[i] = np.square(np.subtract(g_truth[i], input[i])).mean()
        return mse

class Network:
    """This class defines the grid with neurons. One can set the distance between each neuron on the grid."""
    def __init__(self, resolution):

        self.resolution = resolution

        #initializes an empty grid as an 2d array
        self.grid = np.zeros((self.resolution, self.resolution))

        self.neuron_count = 0

        #Initializes the neurons with dist
        self.setNeurons(exp_distance=1)
        print("Grid set up")

    def getNumberofNeurons(self):
        """Returns the total number of neurons there were initialized"""
        return self.neuron_count

    def setNeurons(self,exp_distance):
        """Initializes the grid with neurons. Each neurons gets an id which is stored in the grid"""
        for x in range(self.resolution):
            if x % 2 == 0:
                distance = exp_distance
            else:
                distance = 0

            for y in range(self.resolution):
                if distance == exp_distance:
                    #add neuron
                    self.neuron_count = self.neuron_count + 1
                    self.grid[x][y] = self.neuron_count
                    distance = 0
                else:
                    distance = distance +1

    def getNeurons(self, x_blob, y_blob, receptive_field):
        """Returns the position of the neurons which are active"""
        id_neurons = []
        x_index = int(np.round(x_blob * self.resolution))
        y_index = int(np.round(y_blob * self.resolution))

        x_upper = int(x_index + np.floor(receptive_field/2))
        x_lower = int(x_index - np.floor(receptive_field/2))

        y_upper = int(y_index + np.floor(receptive_field/2))
        y_lower = int(y_index - np.floor(receptive_field/2))


        #check boundary conditions
        if x_upper+1 >= self.resolution:
            x_upper = self.resolution -1

        if y_upper+1 >= self.resolution:
            y_upper = self.resolution -1

        if x_lower < 0:
            x_lower = 0

        if y_lower < 0:
            y_lower = 0


        for x in range(x_lower,x_upper+1):
            for y in range(y_lower,y_upper+1):
                if self.grid[x][y] != 0:
                    id_neurons.append(int(self.grid[x][y]))

        return id_neurons #array with the id of active neurons


    def getNumber(self,x_pos,y_pos):
        """ returns the id of neuron at x_pos, y_pos or 0 if there is not neuron"""
        return self.grid[x_pos][y_pos]
