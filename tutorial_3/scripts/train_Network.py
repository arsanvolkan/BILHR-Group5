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
from layer import *
import matplotlib.pyplot as plt
from tutorial_3.srv import predictJoints, predictJointsResponse


class Model:
    """ Main class which stores the model of the cmac """
    def __init__(self,use_split, use_train_data, receptive_field, resolution, alpha, mode):
        """Main init function-> initializes all important stuff, stores the layer objects in variables
        and switches between train and predict mode"""


        self.receptive_field = receptive_field

        self.resolution = resolution

        if use_train_data:
            #load Data
            self.names, self.data = self.load_data()
          
            #shuffle data
            self.data_shuffled = self.data
            random.shuffle(self.data_shuffled)

            #switch between full and split dataset
            if use_split:
                self.data_shuffled = self.data_shuffled[0:75]

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
            #print(self.data_shuffled)
        else:
            #IF script not started in train mode, start rosserver for prediction
            s = rospy.Service('predict_joints', predictJoints, self.handle_server)
            print("Server ready")
            weights_p, weights_r = self.loadWeights(use_split)

        #define the Network
        self.network = Network(self.resolution)

        #init function says which method is used to init the weights
        # 1 == random | 2 == all weights are 1/12  | else == all weighte are zero
        init_function = 1
        self.layer3 = SimpleLayer(self.network.getNumberofNeurons(), init_function, alpha, mode, receptive_field)

        #in case of predict mode -> load weights
        if not use_train_data:
            self.layer3.load_weights(weights_p, weights_r)

        #MSE Loss
        self.loss_f = MeanSquaredError()

    def handle_server(self,req):
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
        #file = open('C:/Users/bjoer/Documents/bilhr-group5/tutorial_3/scripts/rndmgendata/raw_data.csv')
        csvreader = csv.reader(file)
        header = []
        header = next(csvreader)
        rows = []
        for row in csvreader:
            rows.append(row)
        return header,rows

    def predict(self, blob_x, blob_y):
        """This function uses the trained weights and predicts the angles"""
        max_x = 320.0
        max_y = 240.0
        #normalize between [0,1], maybe hardcode max value and divide
        blob_x = blob_x/max_x
        blob_y = blob_y/max_y

        #call the network
        id_neurons = self.network.getNeurons(blob_x, blob_y, self.receptive_field)
        
        #forward step on the layer with the active neurons as params
        angle_p, angle_r = self.layer3.forward(id_neurons)

        #scale the normalized output of the layer
        angle_p = (angle_p * (0.45+0.19)) - 0.45  # [-2.0857; 2.0857]
        angle_r = (angle_r * (0.5 + 0.32)) - 0.32

        #Make sure its not out of range of the joint limits
        angle_p =  clamp(-2.0857,2.0857,angle_p)
        angle_r = clamp(-0.3142,1.3265,angle_r)

        return angle_p, angle_r #return angle_shoulder_pitch and angle_shoulder_roll

    def train(self,epochs, show_plot):
        """Training functions which iterates over all samples for epoch times"""

        loss_stacked_p = []
        loss_stacked_r = []

        for epoch in range(epochs):
            loss_p = 0
            loss_r = 0
            i = 0
            for sample in self.data_shuffled:
                #the ground truth
                ls_p = sample[0]
                ls_r = sample[1]
                #the training data
                blob_x = sample[2]
                blob_y = sample[3]

                g_truth = np.array((ls_p, ls_r))

                #returns the active neurons
                id_neurons = self.network.getNeurons(blob_x, blob_y, self.receptive_field)

                #computes one forward pass of layer 3 and returns the predicted angles
                # (not in a vector for better readability)
                angle_P, angle_R = self.layer3.forward(id_neurons)

                #make a vector
                out = np.array((angle_P, angle_R))

                #computes the target miss rate for the backward pass: (ground truth  - predicted values)
                ret_miss = np.subtract(g_truth, out)

                #compute the loss
                loss_current = self.loss_f.forward(out,g_truth)

                #stack the loss for one epoch
                loss_p = loss_p + loss_current[0]
                loss_r = loss_r + loss_current[1]

                #backward pass and adjust weights
                self.layer3.backward(loss_current, ret_miss, id_neurons)


            #store for the plot
            loss_p = loss_p/len(self.data_shuffled)
            loss_r = loss_r/len(self.data_shuffled)
            loss_stacked_p.append(loss_p)
            loss_stacked_r.append(loss_r)


        #now plot the loss curve
        fig, ax = plt.subplots()
        epoch_array = np.arange(epochs)
        ax.plot(epoch_array, loss_stacked_p, color="red",label="pitch_error")
        ax.plot(epoch_array, loss_stacked_r, color= "blue", label="roll_error")
        ax.legend(loc="upper left")
        ax.set(xlabel='epochs', ylabel='loss',
        title='Loss curve')
        ax.grid()

        fig.savefig("fig/loss.png")
        if show_plot:
            plt.show()

    def store_model(self,use_split):
        """Calls a function at the layer to save the weights in a .csv file"""
        self.layer3.save_weights(use_split)

    def loadWeights(self, use_split):
        """ loads the trained weights from the .csv file for the predicting mode"""
        path = ''

        if use_split:
            path = './weights/weights_split.csv'
        else:
            path = './weights/weights.csv'

        file = open(path)
        csvreader = csv.reader(file)
        header = []
        header = next(csvreader)
        weights_p = []
        weights_r = []

        #read rows for weights
        weights_p = next(csvreader)
        weights_r = next(csvreader)

        #convert data to float
        weights_p = [float(item) for item in weights_p]
        weights_r = [float(item) for item in weights_r]

        return weights_p, weights_r

def main():
    rospy.init_node('predict_joints_node')
    # instantiate class and start loop function

    """params for the model class"""

    #first for split data (False means dont split)
    use_split = False

    #switch between train mode and prediction mode (True means train), setting this param to False starts an ros service
    
    train_mode =True

    #toggle show plot
    show_plot = True

    #the receptive field of the neurons
    receptive_field = 5

    #the size of the grid
    resolution = 50

    #the learning rate
    alpha = 0.0011

    #defines how the weights are adjusted
    mode = "target_training"#"error_training"#"target_training"#"error_training" #or error_training

    #this is the model class which holds all necessary parts
    model = Model(use_split,train_mode, receptive_field, resolution, alpha, mode)

    if train_mode:
        epochs = 1500

        #let the model train for -epoch- epoches
        model.train(epochs, show_plot)

        if raw_input("Should the weights been stored?(y/n)") == 'y':
            #stores the weights in a csv file
            model.store_model(use_split)
    else:
        """ Just to test the predict function. But should be called via the server"""
        #print(model.predict(0.7,0.8))
        
        rospy.spin()

if __name__=='__main__':
    main()
