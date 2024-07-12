#!/usr/bin/env python

#Group 5
# Laurie Dubois
# Bjoern Doeschl
# Gonzalo Olguin
# Dawen Zhou
# Volkan Arsan


import rospy
import csv
import numpy as np
import random
from network_and_layer import *
import matplotlib.pyplot as plt
from tutorial_4.srv import predictJoints, predictJointsResponse
import time

class Model:
    """ Main class which stores the model of the NN """
    def __init__(self, use_train_data, alpha):
        """Main init function-> initializes all important stuff, stores the network object in a variable
        and switches between train and predict mode"""

        if use_train_data:
            #load Data
            self.names, self.data = self.load_data()

            #shuffle data
            self.data_shuffled = self.data
            random.shuffle(self.data_shuffled)

            #extract columns as vectors
            self.blobx, self.bloby, self.ls_p, self.ls_r = self.extract(self.data_shuffled)

            #normalize data
            self.blobx = self.normalize(self.blobx, "x")
            self.bloby = self.normalize(self.bloby, "y")
            self.ls_p = self.normalize(self.ls_p, "p")
            self.ls_r = self.normalize(self.ls_r, "r")


            #stacks the stuff again to a matrix but with better Accessibility
            self.data_shuffled=[]
            self.data_shuffled=np.column_stack((self.ls_p,self.ls_r))
            self.data_shuffled=np.column_stack((self.data_shuffled,self.blobx))
            self.data_shuffled=np.column_stack((self.data_shuffled,self.bloby))

            #self.writetoTxt(self.data_shuffled)
        else:
            #IF script not started in train mode, start rosserver for prediction
            s = rospy.Service('predict_joints', predictJoints, self.handle_server)
            print("Server ready")
            #loads the weights from the csv file
            weights, bias = self.loadWeights()

        #define the Network(layers are defined in that class)
        self.network = Network(alpha)

        #in case of predict mode -> load weights
        if not use_train_data:
            self.network.load_weights(weights,bias)

        #MSE Loss
        self.loss_f = MeanSquaredError()

        #self.loss_f = CrossEntropyLoss()


    def handle_server(self,req):
        """Computes the response of the server"""
        x = req.x
        y = req.y

        #print("Server called")


        angle_shoulder_pitch, angle_shoulder_roll = self.predict(x,y)

        resp_msg = predictJointsResponse()
        resp_msg.angle1 = angle_shoulder_pitch
        resp_msg.angle2 = angle_shoulder_roll
        return resp_msg

    def extract(self, data):
        """Extract the columns from the matrix and returns them as separate vectors"""
        t_blobx = []
        t_bloby = []
        t_ls_p = []
        t_ls_r = []
        for sample in data:
            t_ls_p.append(float(sample[1]))
            t_ls_r.append(float(sample[2]))
            t_blobx.append(float(sample[3]))
            t_bloby.append(float(sample[4]))
        return t_blobx, t_bloby, t_ls_p, t_ls_r

    def normalize(self, data, var):
        """ Normalizes the x and y data between [0,1]"""
        if var == "x":
            max_ = 320.0
            data = np.dot(data, 1/max_)
        elif var == "y":
            max_ = 240.0
            data = np.dot(data, 1/max_)
        elif var == "p":
            max_ = 2.0857
            min_ = 2.0587
            data = np.dot(np.subtract(data, -0.45), 1/(0.45+0.19))
        elif var == "r":
            max_ = 1.3265
            min_ = -0.3142
            data = np.dot(np.subtract(data, -0.32), 1/(0.5 + 0.32))
        else:
            print("Error in norm")
            data  = []
        return data

    def load_data(self):
        """Loads the data from the csv file and returns it in a matrix"""
        file = open('/home/bio/catkin_ws/src/bilhr-group5/tutorial_3/scripts/data/raw_data_new.csv')

        csvreader = csv.reader(file)
        header = []
        header = next(csvreader)
        rows = []
        for row in csvreader:
            rows.append(row)
        return header,rows

    def writetoTxt(self, data):
        """Writes the normalized data to a txt file for submission"""
        with open('data/data_normed.txt', 'w', newline='') as f:
            f.write("ls_pitch, ls_roll, blob_x, blob_y, \n")
            for row in data:
                for val in row:
                    f.write(str(val))
                    f.write(",")
                f.write("\n")

    def predict(self, blob_x, blob_y):
        """This function uses the trained weights and predicts the angles"""
        max_x = 320.0
        max_y = 240.0

        #normalize between [0,1], maybe hardcode max value and divide
        blob_x = blob_x/max_x
        blob_y = blob_y/max_y

        input = np.array((blob_x, blob_y))
        input = input[...,None]

        #forward step for prediction
        angle_p, angle_r = self.network.forward(input)

        #scale the normalized output of the layer
        angle_p = (angle_p * (0.45+0.19)) - 0.45  # [-2.0857; 2.0857]
        angle_r = (angle_r * (0.5 + 0.32)) - 0.32

        #Make sure its not out of range of the joint limits
        angle_p =  clamp(-2.0857,2.0857,angle_p)
        angle_r = clamp(-0.3142,1.3265,angle_r)

        return angle_p, angle_r #return angle_shoulder_pitch and angle_shoulder_roll

    def train(self,epochs, show_plot):
        """Training functions which iterates over all samples for epoch times"""

        loss_stacked = []
        num = 0
        for epoch in range(epochs):
            loss = 0
            print("Epoch " + str(num))
            for sample in self.data_shuffled:
                #the ground truth
                ls_p = sample[0]
                ls_r = sample[1]

                #the training data
                blob_x = sample[2]
                blob_y = sample[3]

                g_truth = np.array((ls_p, ls_r))
                input = np.array((blob_x,blob_y))
                input = input[...,None]
                g_truth = g_truth[..., None]

                #computes an forward step through the network
                angle_p,angle_r = self.network.forward(input)

                #make a vector
                y_pred = np.array((angle_p, angle_r))

                #computes the target miss rate for the backward pass: (ground truth  - predicted values)
                ret_miss = np.subtract(g_truth, y_pred)

                #compute the loss
                loss_current, e_prime = self.loss_f.forward(y_pred,g_truth)

                #loss backward
                dout = self.loss_f.backward(y_pred)

                #to adjust weights backward through the network
                self.network.backward(dout)

                #stack the loss for one epoch
                loss = loss + loss_current



            #store for the plot
            loss = loss/len(self.data_shuffled)
            loss_stacked.append(loss)
            num = num +1


        #now plot the loss curve
        fig, ax = plt.subplots()
        epoch_array = np.arange(epochs)
        ax.plot(epoch_array, loss_stacked, color="red",label="error")
        ax.legend(loc="upper left")
        ax.set(xlabel='epochs', ylabel='loss',
        title='Loss curve')
        ax.grid()

        fig.savefig("fig/loss.png")
        if show_plot:
            plt.show()

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

def main():
    rospy.init_node('predict_joints_node')
    # instantiate class and start loop function

    """params for the model class"""

    #switch between train mode and prediction mode (True means train), setting this param to False starts an ros service
    train_mode = False

    #toggle show plot
    show_plot = True

    #the learning rate
    alpha = 0.01

    store_weights = True
    #this is the model class which holds all necessary parts
    model = Model(train_mode, alpha)

    if train_mode:
        epochs = 380

        #let the model train for -epoch- epoches
        model.train(epochs, show_plot)

        if store_weights:
            #stores the weights in a csv file
            model.store_model()
    else:
        """ Just to test the predict function. But should be called via the server"""
        #print(model.predict(200,100))

        rospy.spin()

if __name__=='__main__':
    main()
