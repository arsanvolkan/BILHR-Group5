#!/usr/bin/env python

#Group 5
# Laurie Dubois
# Bjoern Doeschl
# Gonzalo Olguin
# Dawen Zhou
# Volkan Arsan

#allowing rospy to start if true and also services to load
in_lab =False
if in_lab:
    import rospy
    from tutorial_5.srv import *
    from std_msgs.msg import *
    from std_srvs.srv import *

import csv
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time
from sklearn import tree
import copy

class Model:
    """Stores all the variables and all RL stuff is implemented in here"""
    def __init__(self, Rmax):
        """To initialize the values"""
        self.actions = {'Move_in':0,'Move_out':1,'Kick':2} #actions can be [0==Move_in, 1==Move_out, 2==Kick] when facing the robot, move in means move left, and move out is move right
        self.number_of_states = 11
        self.number_of_goalkeeperstates = 4
        start_state_index = 5
        self.goalkeeperstate = 3#self.getGoalKeeperState()
        self.rmax = Rmax
        self.MAX_STEPS = 5 #
        self.max_neg_reward = -10000
        self.sm = []
        self.sm.append(start_state_index)
        self.states_exact = np.linspace(0.25, 0.75,self.number_of_states) #not used

        self.current_state_index = start_state_index
        self.pm = []
        self.rm = []
        self.gamma = 0.9
        self.q_table = []
        self.q_table_compare=[]
        # 0 for not converged, 1 for converged
        self.convergetable=[]


        self.rewards_tree = None
        self.transition_tree = None

        self.x_reward = np.array([[]])
        self.y_reward = np.array([])

        self.x_trans = np.array([[]])
        self.y_trans = np.array([])

        self.visits = [] #not used

        self.initTables()
        # to make a copy,
        #  if not only pointer is given
        self.q_table_compare=copy.deepcopy(self.q_table)
        self.q_table_old = copy.deepcopy(self.q_table)




    def initTables(self):
        """This functions inits all our lookup tables"""
        for i in range(self.number_of_states):
            q_matrix = []
            pm_matrix = []
            rm_matrix = []
            visits_matrix = []
            for gks in range(self.number_of_goalkeeperstates):
                q_matrix.append([0.0,0.0,0.0])
                pm_matrix.append([0.0,0.0,0.0])
                rm_matrix.append([0.0,0.0,0.0])
                visits_matrix.append([0.0,0.0,0.0])


            self.q_table.append(q_matrix)
            self.pm.append(pm_matrix)
            self.rm.append(rm_matrix)
            self.visits.append(visits_matrix)
        
        for s in range(self.number_of_goalkeeperstates):
            self.convergetable.append(0)

    # this function is used in compute values, to check if the current update value should be wright in the qtable
    # True means in current goal keeper state, values have already converged, and there is no need to training in this goal keeper state
    # in this case the user should pull the goal keeper in a new state
    # False means in current goal keeper state values needs to be updated and so on
    def checkIfConvergeInstates(self,exploration):
        # check goal keeper state first, if one state converge then maybe print a line and get out the training loop
        if self.convergetable[self.goalkeeperstate]==1:
            return True
        else:
            gks=self.goalkeeperstate
            for i in range(self.number_of_states):
                for action in range(3):
                    if self.q_table[i][gks][action]-0.01>self.q_table_compare[i][gks][action] or self.q_table_compare[i][gks][action]>self.q_table[i][gks][action]+0.01:
                       return False

            if exploration:
                return False
            self.convergetable[self.goalkeeperstate]=1
            return True


    def checkIfConvergeALL(self):
        for i in self.convergetable:
            if i==0:
                return False
        return True

    def addExpToRewardTree(self, state, gks, action, reward):
        """Adds the new state goalkeeper action pair with the reward as label"""
        if self.rewards_tree==None:
            #if no tree exists yet
            self.x_reward = np.array([[state, gks, action]])
            self.y_reward = np.array([reward])
            clf = tree.DecisionTreeClassifier()
            clf.fit(self.x_reward,self.y_reward)
            self.rewards_tree = clf
            ch=True
            print("State: " +str(state) + " Action: " +str(action)+ " Reward: "+ str(reward))
        else:
            if  self.checkIsInArray(self.x_reward, self.y_reward, state, gks, action, reward):
                #the state action pair is already in the table
                ch=False
            else:
                self.x_reward = np.append(self.x_reward, [[state, gks, action]], axis=0)
                self.y_reward = np.append(self.y_reward, [reward], axis=0)
                clf = tree.DecisionTreeClassifier()
                clf.fit(self.x_reward,self.y_reward)
                self.rewards_tree = clf
                print("State: " +str(state) + " Action: " +str(action)+ " Reward: "+ str(reward))
                ch=True
        return ch





    def addExperienceToTransitionTree(self, state, gks, action, transition):
        """Adds the new state goalkeeper action pair with the transistion to the tree"""
        if self.transition_tree==None:
            #if no tree exists yet
            self.x_trans = np.array([[state, gks, action]])
            self.y_trans = np.array([transition])
            clf = tree.DecisionTreeClassifier()
            clf.fit(self.x_trans,self.y_trans)
            self.transition_tree = clf
            ch=True
        else:
            if self.checkIsInArray(self.x_trans,self.y_trans, state, gks, action, transition):
                #the state action pair is already in the table
                ch=False
            else:
                self.x_trans = np.append(self.x_trans, [[state, gks, action]], axis=0)
                self.y_trans = np.append(self.y_trans, [transition], axis=0)
                clf = tree.DecisionTreeClassifier()
                clf.fit(self.x_trans,self.y_trans)
                self.transition_tree = clf
                ch=True
        return ch

    def checkIsInArray(self, nparray, y_label, state, gks, action, value):
        """Checks if the given pair is in the array already or not"""
        isinarray = np.isin(nparray, [[state, gks, action]])
        isinarray_y = np.isin(y_label, [value])
        for i in range(len(isinarray)):
            if isinarray[i][0] == True and isinarray [i][1] == True and isinarray [i][2] == True and isinarray_y[i] == True:
                return True
        return False

    def update_model(self, state, action, reward, new_state, sm, action_set):
        """Updates the trees and gets the predictions for transisiton and rewards"""
        pm = self.pm
        rm = self.rm
        gks = self.goalkeeperstate
        ch = False

        x = new_state - state

        #Here the tree is updated
        ch1 = self.addExperienceToTransitionTree(state, gks, action, x)

        ch2 = self.addExpToRewardTree(state, gks, action, reward)

        #checks if the model has changed
        if ch1==True or ch2==True:
            ch=True

        for s in self.sm:
            for a in action_set:
                #here product of all trees
                pair = np.array([[s, gks, a]])
                total_x = self.transition_tree.predict(pair)
                s_total = s + total_x
                pm[s][gks][a] = self.combineResults(s, gks, a, state, action)
                if s == state and a == action:
                    rm[s][gks][a] = self.getPredictions(s, gks, a, state, action)
        return pm, rm, ch

    def combineResults(self, state, gks, action, state_, action_):
        #should iterate over trees to combine the state trans prediction but it is one in our case
        return 1

    def getPredictions(self, state, gks, action, state_act, action_act):
        """Just returns the predictions from the tree"""
        r = 0
        pair = np.array([[state, gks, action]])
        r = float(self.rewards_tree.predict(pair))
        return r

    def checkState(self,state):
        """Checks if the new state is out of range"""
        if state < 0:
            return -5
        elif state >= self.number_of_states:
            return -5
        else:
            return state

    def evaluate_bellman(self):
        """Evaluates the bellman equation for the optimal policy"""
        q = []
        state = self.current_state_index
        gks = self.goalkeeperstate

        for a in range(len(self.actions.values())):
            pair = np.array([[state, gks, a ]])
            next_state = self.getNewState(state,a)
            next_state = self.checkState(next_state)
            if next_state == -5:
                q_next = self.max_neg_reward
            else:
                q_next = max(self.q_table[next_state][gks])
            #print("q_next: "+str(q_next))
            #since only one state is possible no summation here
            sum_ = 1.0*q_next
            #print("sum: "+str(sum_))
            #q_ = float(self.rewards_tree.predict(pair)) + self.gamma * sum_
            q_ = float(self.rm[state][gks][a]) + self.gamma * sum_
            #print("q_: " +str(q_))

            q.append(q_)

        return q

    def getAction(self):
        """Returns action with highest prob"""
        #action = from look up table
        if self.rewards_tree == None:
            print("Tree not initialized. Selecting random action.")
            action = random.choice(list(self.actions.values()))
        else:
            q = self.evaluate_bellman()
            print(q)
            action = 0
            max=np.argmax(q)
            if q[0] == q[1] and q[1]==q[2]:
                action = random.choice(list(self.actions.values()))
                print("All qs are the same. Choosing random action!")

            elif q[0] == q[1] and q[1]==q[max]:
                action = random.choice([0,1])
            elif q[1] == q[2] and q[1]==q[max]:
                action = random.choice([1,2])
            elif q[0] == q[2] and q[0]==q[max]:
                action = random.choice([0,2])
            else:
                action = max
            print("Action is: " + str(action))
        return action

    def computeValues(self, rmax, pm, rm, sm, actions, exp):
        """Computes the new action values is from the paper adapted"""
        #init the K steps to neighbour
        gks = self.goalkeeperstate
        k = []
        for s in sm:
            #print("s:" +str(s))
            exist_action = False
            for a in actions:
                if self.visits[s][gks][a] > 0:
                    k.append(0)
                    exist_action = True
                    break
            if not exist_action:
                k.append(float('inf'))
        #print("K: " +str(k))
        #print("__________________")

        minvisits = float('inf')
        for s in range(self.number_of_states):
            min_a = min(self.visits[s][gks])
            if minvisits > min_a:
                minvisits = min_a
        #print("Min visti:  " +str(minvisits))
        div = False
        #do while not diverged
        sm_iter = [elem for elem in sm]
        i = 0
        #div == False
        while div == False:

            for s in self.sm:#maybe change to sm
                for a in actions:
                    #print(s)
                    #print(k)
                    #print(a)

                    #for all states explored and actions do update q values
                    if exp and self.visits[s][gks][a]==minvisits:
                        #unknown states exploration bonus
                        self.q_table[s][gks][a] = rmax
                        self.q_table_old[s][gks][a] = rmax
                        div = True
                        i = 10
                        #print("First")
                        continue
                    elif k[self.sm.index(s)] > self.MAX_STEPS:
                        #these states are out of reach
                        self.q_table[s][gks][a] = rmax
                        self.q_table_old[s][gks][a] = rmax
                        div = True
                        i = 10
                        #print("Second")
                        continue
                    else:
                        #update
                        #print("Third")
                        self.q_table[s][gks][a] = rm[s][gks][a]
                        #dont iterate over actions because only one action
                        #for a_ in actions:
                            #compute s'
                        if a == 0:
                            x = -1
                        elif a == 1:
                            x = 1
                        elif a == 2:
                            x = 0
                        else:
                            print("Error in computeValues")
                            return
                        s_ = s + x
                        if s_ < 0 or s_ >= self.number_of_states:
                            pass
                        else:
                            if s_ not in self.sm:
                                #add to visited states
                                self.sm.append(s_)
                                k.append(float('inf'))
                                #visit_total.append(0)

                            #Update steps to this state
                            #print("s_= "+ str(s_))

                            if k[self.sm.index(s)] + 1 < k[self.sm.index(s_)]:
                                k[self.sm.index(s_)] = k[self.sm.index(s)] + 1
                            #here the P(s'|s,a) is one because there is only one possible next state and the others two actions with P = 0 are not considered.

                            equ = self.gamma * 1.0 * max(self.q_table[s_][gks])
                            self.q_table[s][gks][a] = self.q_table[s][gks][a] + equ



                    #print("----------------")
            for i in range(self.number_of_states):
                for a in actions:
                    diff = math.fabs(self.q_table_old[i][gks][a] - self.q_table[i][gks][a])
                    if diff > 0.001:
                        div = False
                        self.q_table_old = copy.deepcopy(self.q_table)
                        continue
                    else:
                        div = True
            self.q_table_old = copy.deepcopy(self.q_table)

            i = i +1
        print("Updated Values")
        return 0

    def checkPolicy(self, pm, rm,reward, reward_stacked):
        """Checks whether there is a state in the explored set which has only negative or only positive reward"""
        exp = False

        rew_max = -2
        gks = self.goalkeeperstate
        for s in self.sm:
            for a in list(self.actions.values()):
                rew_max=max(rew_max,rm[s][gks][a])

        if rew_max<8:
            # explore
            exp = True
        else:
            # exploit
            exp = False

        return exp

    def getNewState(self, state_index, action):
        """Transistion function... returns the new state for the action"""
        if action==0:
            return state_index -1
        elif action==1:
            return state_index+1
        elif action==2:
            return state_index
        else:
            print("Error this action is not coded!")

    def execute(self):
        """Executes the best actions according to the optimal policy to score a goal"""
        state = 5
        action = 0
        while action != 2:
            gks = self.getGoalKeeperState()
            print("Goalkeeper at: "+ str(gks))
            action = self.getExecuteAction(state, gks)
            self.executeAction(action)
            state_new = self.getNewState(state,action)
            print("At state: "+str(state)+" doing action: "+str(action)+" leading to new state: "+ str(state_new))
            state = state_new
            if state_new == -1:
                print("Policy not good")
                return
            elif state_new >= self.number_of_states:
                print("Policy not good 2")
                return

        print("Scored a goal yeahhhhhh!!!")


    def getExecuteAction(self, state, gks):
        """This function returns the best action according to the optimal policy"""
        max_rew = -5
        action = -1

        for a in list(self.actions.values()):
            pair = np.array([[state, gks, a ]])
            rew = self.rewards_tree.predict(pair)
            rew_pred = self.rewards_tree.predict_proba(pair)
            if rew > max_rew:
                max_rew = rew
                action = a

        return action

    def loadTree(self):
        """builds the tree wioth the loaded arrays"""
        self.x_reward, self.y_reward = self.loadfromCSV()
        #print(self.x_reward, self.y_reward)
        clf = tree.DecisionTreeClassifier()
        clf.fit(self.x_reward,self.y_reward)
        self.rewards_tree = clf

    def loadfromCSV(self):
        """Loads the optimal policy from the csv file"""
        path = './weights/x_rew_.csv'
        file = open(path)
        csvreader = csv.reader(file)
        header = []
        header = next(csvreader)
        row = next(csvreader)
        x_rew = np.array([])

        while row != ["end"]:
            if row != ["next"]:
                if x_rew.size == 0:
                    row = [int(item) for item in row]
                    x_rew = np.array([row])
                    row = next(csvreader)
                else:
                    row = [int(item) for item in row]
                    x_rew = np.append(x_rew, [row], axis=0)
                    row = next(csvreader)
            else:
                row = next(csvreader)

        path = './weights/y_rew_.csv'
        file = open(path)
        csvreader = csv.reader(file)
        header = []
        header = next(csvreader)
        row = next(csvreader)
        y_rew = np.array([])

        while row != ["end"]:
            if row != ["next"]:
                if y_rew.size == 0:
                    y_rew = np.array([int(row[0])])
                    row = next(csvreader)
                else:
                    y_rew = np.append(y_rew, [int(row[0])], axis=0)
                    row = next(csvreader)
            else:
                row = next(csvreader)
        return x_rew, y_rew


    def savetoCSV(self):
        """Saves the reward tree arrays to a csv file. So the optimal policy is stored"""
        print(self.x_reward, self.y_reward)
        header=["x_rew"]
        name = './weights/x_rew.csv'
        with open(name, 'w') as outfile:#, newline='') as outfile:
            writer=csv.writer(outfile)
            writer.writerow(header)
            for row in self.x_reward:
                writer.writerow(row)
                writer.writerow(["next"])
            writer.writerow(["end"])
        header=["y_rew"]
        name = './weights/y_rew.csv'
        with open(name, 'w') as outfile:   #, newline='') as outfile:
            writer=csv.writer(outfile)
            writer.writerow(header)
            for row in self.y_reward:
                writer.writerow([int(row)])
                writer.writerow(["next"])
            writer.writerow(["end"])
        print("done writing file, all weights have been written")


    def train(self, show_plot, epoch):
        """Train methods performs the train till convergence. after training is done, the reward tree will be stored"""

        reward_stacked = []
        reward_add = 0
        ##########################
        action = -1 #for example
        i = 0
        while(i < epoch):
            try:
                if self.checkIfConvergeALL():
                    print("!!!!!!!!!!!!!training in all goal keeper states are done!!!!!!!!!!!!!!!!!!!")
                    break
                #get action with argmax(q)
                self.goalkeeperstate = self.getGoalKeeperState()
                print("Goalkeeper at: "+ str(self.goalkeeperstate))
                #get action with argmax(q)
                action = self.getAction()
                #computes the new state if action is executed
                new_state_index = self.getNewState(self.current_state_index, action)
                print("State: "+str(self.current_state_index) + " and Action: "+str(action) + " leading to new State: "+ str(new_state_index))


                #checks if the new state is out of bounds and return a bad reward for this case
                if new_state_index < 0 or new_state_index >= self.number_of_states:
                    print("Error out of bounds")
                    reward = self.max_neg_reward
                    violated = True
                    new_state_index = self.current_state_index
                    reward_add += reward
                    reward_stacked.append(reward_add)
                    break
                else:
                    #execute the action and get the reward
                    reward = self.executeAction(action)
                    reward_add += reward
                    reward_stacked.append(reward_add)
                    #increments the state action visit in the look up table
                    self.visits[self.current_state_index][self.goalkeeperstate][action] = self.visits[self.current_state_index][self.goalkeeperstate][action] + 1

                    if new_state_index not in self.sm:
                        self.sm.append(new_state_index)

                    #update the model
                    self.pm, self.rm, ch = self.update_model(self.current_state_index, action, reward, new_state_index, self.sm, list(self.actions.values()) )
                    #check the policy
                    #print("RM")
                    #print(self.rm)
                    #print("-----------------------------------")
                    exp = self.checkPolicy(self.pm, self.rm, reward, reward_stacked)
                    print("Exploration: "+str(exp))
                    if ch:
                        #if model changed recompute the Q values
                        self.computeValues(self.rmax, self.pm, self.rm, self.sm, list(self.actions.values()), exp)
                        print("changed")
                    if self.checkIfConvergeInstates(exp):
                        # only gives hint, no need to end training
                        print("Goalkeeper in state "+str(self.goalkeeperstate)+" has already converged. Move goal keeper !!")

                    else:
                        #update q_table_compare
                        print(self.convergetable)
                        self.q_table_compare=copy.deepcopy(self.q_table)
                    self.current_state_index = new_state_index
                    print("Epoch:"+ str(i))
                    i += 1


            except KeyboardInterrupt:
                break


        self.savetoCSV()
        ##########################
        #now plot the loss curve
        fig, ax = plt.subplots()
        ax.plot(reward_stacked, color="red",label="reward")
        ax.legend(loc="upper left")
        ax.set(xlabel='iterations', ylabel='reward',
        title='Reward curve')
        ax.grid()

        fig.savefig("fig/reward_stacked.png")
        if show_plot:
            plt.show()

    def getGoalKeeperState(self):
        """Calls the service goalkeeperstate to retrieve the current goal keeper state"""
        state = 0
        if in_lab:
            rospy.wait_for_service('getGoalKeeperState')
            try:
                execute = rospy.ServiceProxy('getGoalKeeperState', goalkeeperstate_msg)
                request = 0
                response = execute(request)
                state = response.state
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
        else:
            state = 4#int(input("GOAL KEEPER STATE: 0, 1 , 2, 3 or no goal keeper is 3:   "))
            if state > self.number_of_goalkeeperstates-1:
                state = self.number_of_goalkeeperstates-1
            elif state < 0:
                state = 0
        return state

    def executeAction(self, action):
        """If in_lab is false here send action to robot and wait for reward as return.
        if not in_lab then input the reward by hand and dont execute on robot"""

        if in_lab:

            rospy.wait_for_service('executeAction')
            try:
                reward = 0
                execute = rospy.ServiceProxy('executeAction', reward_msg)
                response = execute(action)
                reward = response.reward

                print("Rewardservice:"+ str(response.reward))
                print("Rewardservice:"+ str(reward))
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
                reward = 0

        else:
            reward = int(input("Reward: (-1->for each action || -20->fell down || +20->goal || -2->notgoalkicked): "))

        return reward

def main():
    if in_lab:
        rospy.init_node('reinforcement_learning_node')
    # instantiate class and start loop function

    """params for the model class"""

    #toggle show plot
    show_plot = True

    train_mode = True

    rmax = 1.0
    epochs = 100000

    #this is the model class which holds all necessary parts
    model = Model(rmax)
    if train_mode:
        model.train(show_plot, epochs)

    else:
        model.loadTree()
        model.execute()
    print("Done")

    if in_lab:
        rospy.spin()

if __name__=='__main__':
    main()
