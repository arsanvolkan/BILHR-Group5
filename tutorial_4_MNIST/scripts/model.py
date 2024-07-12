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
from network_and_layer import *
import matplotlib.pyplot as plt
from mnist import MNIST
import time

class Model:
    """ Main class which stores the model of the NN """
    def __init__(self, use_train_data, alpha):
        """Main init function-> initializes all important stuff, stores the network object in a variable
        and switches between train and predict mode"""

        #load Data of the MNIst data set
        mndata = MNIST('./../data')
        mndata.gz = True
        (self.images, self.labels) = mndata.load_training()


        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.images = self.normalize(self.images)
        length = self.images.shape[0]

        #divide data into test validation and train
        self.train_data = self.images[0:36000]

        self.val_data = self.images[36001:48000]
        self.val_labels = self.labels[36001:48000]


        self.test_data = self.images[48001:-1]
        self.test_labels = self.labels[48001:-1]

        if use_train_data:
            pass
        else:
            #IF script not started in train mode, loads weights and checks the network via validation set
            print("Predicting Mode")
            weights, bias = self.loadWeights()

        #define the Network
        self.network = Network(alpha)

        #in case of predict mode -> load weights
        if not use_train_data:
            self.network.load_weights(weights,bias)

        #MSE Loss
        #self.loss_f = MeanSquaredError()

        self.loss_f = CrossEntropyLoss()

    def normalize(self, data):
        """ Normalizes the x and y data between [0,1]"""
        data = np.dot(data, 1/255.0)
        return data

    def predict(self):
        """do a forward step through the network and use the pretrained weights"""

        #to store the accuracy
        acc = 0.0

        #right positive
        rp = 0
        i = 0
        for image in self.val_data:
            image_ = image[...,None]
            y_pred = self.network.forward(image_)

            label = np.argmax(y_pred)

            if self.val_labels[i] == label:
                rp = rp+1
            i = i+1
            help = self.toMatrix(image,28)
            im = np.array(help)
            #fig = plt.figure
            #plt.imshow(im, cmap='gray')
            #plt.show()

        acc = float(rp)/float(self.val_data.shape[0])

        print("Acc: "+ str(acc))
        return label


    def toMatrix(self, data_vector, size):
        """From vector to image matrix"""
        matrix = []
        for x in range(size):
            row = []
            for i in range(size):
                row.append(data_vector[i+ x *28])
            matrix.append(row)
        return matrix

    def store_model(self):
        """Calls a function at the layer to save the weights in a .csv file"""
        self.network.save_weights()

    def loadWeights(self):
        """ loads the trained weights from the .csv file for the predicting mode"""
        path = ''
        path = './weights/weights.csv'

        file = open(path)
        csvreader = csv.reader(file)
        header = []
        header = next(csvreader)
        weight_tensor = []
        bias_matrix = []
        row = next(csvreader)
        weight = []
        bias = []
        switch = "weights"

        while row != ["end"]:
            if type(row[0]) == type(""):
                if row == ["next"]:
                    switch = "bias"
                    row = next(csvreader)
                elif row == ["next_b"]:
                    switch = "store"
                    row = next(csvreader)

            if switch == "weights":
                #now read weights of this layers
                row = [float(item) for item in row]
                weight.append(row)
                row = next(csvreader)

            elif switch == "bias":
                row = [float(item) for item in row]
                bias.append(row)
                row = next(csvreader)

            elif switch == "store":
                #convert data to float
                weight_tensor.append(weight)
                bias_matrix.append(bias)
                weight = []
                bias = []
                switch = "weights"
            else:
                print("Error in weigth loader")


        return weight_tensor, bias_matrix

    def train(self,epochs, show_plot):
        """Training functions which iterates over all samples for epoch times"""
        print("Start Training")
        num = 0
        loss_stacked = []
        acc_stacked = []
        for epoch in range(epochs):
            print("Epoch "+ str(num))
            loss = 0
            i = 0
            acc = 0
            for sample in self.train_data:
                #the ground truth
                g_t = self.labels[i]
                #to one hot vector
                one_hot_vector = np.zeros(10)
                one_hot_vector[g_t] = 1
                one_hot_vector = one_hot_vector[...,None]
                sample = sample[...,None]

                #computes an forward step through the network
                y_pred = self.network.forward(sample)
                label = np.argmax(y_pred)
                if g_t == label:
                    acc = acc+1

                #compute the loss
                loss_current, e_prime = self.loss_f.forward(y_pred, one_hot_vector)

                #stack the loss for one epoch
                loss = loss + loss_current

                ret_miss = np.subtract(y_pred, one_hot_vector)

                #computes the target miss rate for the backward pass: (ground truth  - predicted values)
                dout = self.loss_f.backward(loss)

                #to adjust weights
                self.network.backward(ret_miss)

                i = i+1
            num = num +1

            #store for the plot
            loss = loss/len(self.train_data)
            acc = acc/float(len(self.train_data))
            loss_stacked.append(loss)
            acc_stacked.append(acc)


        #now plot the loss curve
        fig, ax = plt.subplots()
        epoch_array = np.arange(epochs)
        ax.plot(epoch_array, loss_stacked, color="blue",label="error")
        ax.plot(epoch_array, acc_stacked, color="red",label="acc")
        ax.legend(loc="upper left")
        ax.set(xlabel='epochs', ylabel='loss',title='Loss curve')
        ax.grid()

        fig.savefig("fig/loss.png")
        if show_plot:
            plt.show()


def main():
    # instantiate class and start loop function
    """params for the model class"""

    #switch between train mode and prediction mode (True means train), setting this param to False starts an ros service

    train_mode = False

    #toggle show plot
    show_plot = True

    #the learning rate
    alpha = 0.001

    store_weights = False
    #this is the model class which holds all necessary parts
    model = Model(train_mode, alpha)

    if train_mode:
        epochs = 10
        start = time.time()
        #let the model train for -epoch- epoches
        model.train(epochs, show_plot)
        ende = time.time()

        print('Total time: {:5.3f}min'.format(((ende-start)/60.0)))

        if store_weights:
            #stores the weights in a csv file
            model.store_model()
    else:
        """ Just to test the predict function. Runs the validation set"""
        model.predict()


if __name__=='__main__':
    main()
