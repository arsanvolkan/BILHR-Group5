#!/usr/bin/env python

#Group 5
# Laurie Dubois
# Bjoern Doeschl
# Gonzalo Olguin
# Dawen Zhou
# Volkan Arsan

from random import random
import rospy

from std_msgs.msg import *
from std_srvs.srv import *
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import json
import csv
from tutorial_5.srv import *



class Central:

    def __init__(self):
        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.blob_pos_x = 0
        self.blob_pos_y = 0
        self.jointPub = 0
        self.stiffness = False

        #number of states to map the deviation to the states
        self.number_of_states = 11
        self.step_size = 0.5/(self.number_of_states-1)

        # goal keeper related parameters
        self.goalkeeperstatesNumber=3  #from left to right 0,1,2,3,4,5
        self.leftdoorposition=0
        self.rightdoorposition=0
        self.goalkeeperposition=0
        self.goalkeeperstate=3



        #here the reward is stored for the return
        self.reward = 0
        print("Init")

        pass


    def key_cb(self,data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def joints_cb(self,data):
        # store current joint information in class variables
        self.joint_names = data.name
        self.joint_angles = data.position
        self.joint_velocities = data.velocity



    def bumper_cb(self, data):
        rospy.loginfo("bumper: "+str(data.bumper)+" state: "+str(data.state))
        if data.bumper == 0:
            self.stiffness = True
        elif data.bumper == 1:
            self.stiffness = False


    # This function is activated when the buttons are pressed
    def touch_cb(self,data):
        rospy.loginfo("touch button: "+str(data.button)+" state: "+str(data.state))

        if data.button == 1 and data.state == 1:
            #reward
            self.kick_action()
            pass

        elif data.button == 2 and data.state==1:
            self.set_stiffness(False)
            pass


        elif data.button == 3 and data.state == 1:
            self.setToHomePosition()


    def handle_goalserver(self, req):
        """This function gets an virtual value which is not used, but returns the current goalkeeper position to the RL alg"""
        x = req.ask
        response = goalkeeperstate_msgResponse()
        print("GKS: "+ str(self.goalkeeperstate))
        response.state = int(self.goalkeeperstate)
        return response

    def handle_server(self, request):
        """gets the action via the server, executes the action [0,1,2] and waits for a reward input and returns this reward"""
        action = int(request.action)

        if action==0:
            self.move_in()
        elif action==1:
            self.move_out()
        elif action==2:
            self.kick_action()
        else:
            print("Weird things happend")

        train_mode = True
        if train_mode == True:
            reward = 0
            reward = int(raw_input("Enter reward:"))
            print("Reward is "+ str(reward))
        else:
            reward = 0
        response = reward_msgResponse()
        response.reward = reward

        return response


    def move_in(self):
        """For action 0: Moves the leg one state to the inside"""
        print("Move in")
        LHipRoll = self.joint_angles[9]
        LHipRoll -= self.step_size
        self.set_joint_angles(["LHipRoll"],[LHipRoll])

    def move_out(self):
        print("Move out")
        """For action 1: Moves the leg one state to the outside by the specified step size"""
        LHipRoll = self.joint_angles[9]
        LHipRoll += self.step_size
        self.set_joint_angles(["LHipRoll"],[LHipRoll])


    # This function is to get the raw images from the NAO's top camera
    # then convert them in HSV.
    # It applies a filter on the image with masks and converts red to black and
    # everything else to white.
    # Erosion and dilation is used to smoothen the image.
    # Then the blob detector detects the black pixels and returns them in blolbs.
    # Finally, a green circle is drawn around the red shape, and we extract the coordinates.
    def image_cb(self,data):
        bridge_instance = CvBridge()
        try:
            image = bridge_instance.imgmsg_to_cv2(data, "bgr8")
            image = cv2.resize(image, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

            arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
            arucoParams = cv2.aruco.DetectorParameters_create()
            (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
        except CvBridgeError as e:
            rospy.logerr(e)

        winname = "display"
        cv2.namedWindow(winname)  # Create a named window
        cv2.moveWindow(winname, 400, 300)
        cv2.imshow(winname, image)
        cv2.waitKey(3)


        if len(corners) > 0:
            #rUCo ID
            ids = ids.flatten()
            # ArUCo
            #print(ids)
            if len(ids) >2:
                for (markerCorner, markerID) in zip(corners, ids):
                    # TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, _, bottomRight, _) = corners
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    if markerID==1:
                        self.leftdoorposition = int((topLeft[0] + bottomRight[0]) / 2.0)
                        #print("id "+ str(markerID)+" central position is:"+ str(self.leftdoorposition ))

                    elif markerID==2:
                        self.rightdoorposition = int((topLeft[0] + bottomRight[0]) / 2.0)
                        #print("id "+ str(markerID)+" central position is:"+ str(self.rightdoorposition))
                    else:
                        self.goalkeeperposition=int((topLeft[0] + bottomRight[0]) / 2.0)
                        #print("id "+ str(markerID)+" central position is:"+ str(self.goalkeeperposition)
                    cv2.imshow(winname, image)
                    cv2.waitKey(1)
            
                self.goalkeeperstate=self.getGoalKepperState()
            else:
                self.goalkeeperstate = 3
        else:
            self.goalkeeperposition = 3#self.goalkeeperstatesNumber
        #print("Current goal keeper is in state:"+str(self.goalkeeperstate))


    def getGoalKepperState(self):
        # door length is X1_left-X1_right can be different because of the manually configeration,
        # so we should only feed in goal keeper states to the RL every time
        # we can use another marker with other ids, as goal keeper,
        # with maybe 4 states using a box as goalkeeper?
        if self.leftdoorposition!=0 and self.rightdoorposition!=0 and self.goalkeeperposition!=0:
            eachstepsize=(self.rightdoorposition-self.leftdoorposition)/self.goalkeeperstatesNumber
            return min( max((self.goalkeeperposition-self.leftdoorposition)//eachstepsize,0),self.goalkeeperstatesNumber-1)
        else:

            return 3



    def setToKickPosition(self):
        """Sets the roboter to the Kick position. Standing upright"""
        self.set_joint_angles(["HeadYaw","HeadPitch","LHand"],[0.3622,-0.105,0])
        rospy.sleep(2)

    def set_stiffness_Task(self):
        """Set the stiffness for the kick"""
        effort = [1]*26
        effort[0] = 0.9
        effort[1] = 0.9
        effort[2] = 0
        effort[3] = 0


        self.set_stiffness_single(effort)


    def set_stiffness_single(self,effort):
        """Sets the stiffness to the values in the effort vectors"""

        jointStiffness = JointState()
        ratePub = rospy.Rate(5)
        #need to publish 5 times to remember the stiffness
        i = 0
        while i < 5:
            jointStiffness = JointState()
            jointStiffness.effort = effort
            jointStiffness.name = ["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
            "LHand", "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll", "RHipYawPitch",
            "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll", "RShoulderPitch", "RShoulderRoll",
            "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"]
            self.stiffPub.publish(jointStiffness)
            ratePub.sleep()
            i = i +1
        pass



    # sets the stiffness for all joints. can be refined to only toggle single joints, set values between [0,1] etc
    def set_stiffness(self,value):
        if value == True:
            service_name = '/body_stiffness/enable'
        elif value == False:
            service_name = '/body_stiffness/disable'
        try:
            stiffness_service = rospy.ServiceProxy(service_name, Empty)
            stiffness_service()
        except rospy.ServiceException as e: #it said ,e
            rospy.logerr(e)

    # this function is to be able to move several joints at the same time
    # it calls the given function to control the joints
    def set_joint_angles(self,joint_names, head_angles):

        joint_angles_to_set = JointAnglesWithSpeed()
        for i in range(len(joint_names)):
            # The for loop is so that multiple joint angles can be published at once
            # and the movements are simultaneous and not sequential
            joint_angles_to_set.joint_names.append(joint_names[i]) # each joint has a specific name, look into the joint_state topic or google
            joint_angles_to_set.joint_angles.append(head_angles[i]) # the joint values have to be in the same order as the names!!

        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.1 # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)


    def set_joint_angles_with_speed(self,joint_names, head_angles,speed):

        joint_angles_to_set = JointAnglesWithSpeed()
        for i in range(len(joint_names)):
            joint_angles_to_set.joint_names.append(joint_names[i]) # each joint has a specific name, look into the joint_state topic or google
            joint_angles_to_set.joint_angles.append(head_angles[i]) # the joint values have to be in the same order as the names!!

        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = speed # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)




    def setToHomePosition(self):
        self.set_stiffness(True)
        name_all_joints=["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
            "LHand", "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll", "RHipYawPitch",
            "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll", "RShoulderPitch", "RShoulderRoll",
            "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"]

        values_all_joints=[0.018366098403930664, 0.30675792694091797, 1.9987601041793823, 1.3023240566253662,
                            -0.6151759624481201, -0.03490658476948738, -1.8238691091537476, 0.42320001125335693,
                            -0.13955211639404297, 0.50, 0.38814401626586914, 0.3,
                            -0.0, -0.3,-0.18250393867492676, 0.13810205459594727,
                            0.47089600563049316, -0.08432793617248535, -0.10426998138427734, -0.3021559715270996,
                            1.8208999633789062, -0.76857590675354, 1.21642005443573, 0.4065520763397217, 0.9203581809997559, 0.8531999588012695]#0.50264605522155762
        self.set_joint_angles(name_all_joints,values_all_joints)
        rospy.sleep(2)
        # #[0.25;0.75]
        # self.set_joint_angles(["LHipRoll"],[0.75])
        # rospy.sleep(2)
        # self.set_joint_angles(["LHipRoll"],[0.55])
        # rospy.sleep(2)
        # self.set_joint_angles(["LHipRoll"],[0.45])
        # rospy.sleep(2)
        # self.set_joint_angles(["LHipRoll"],[0.35])
        # rospy.sleep(2)
        # self.set_joint_angles(["LHipRoll"],[0.25])
        # rospy.sleep(2)


    def kick_action( self):
        """Performs the kick"""
        print("Kick")
        self.set_stiffness(True)
        # this action is for leg going back
        self.set_joint_angles_with_speed(['LKneePitch'],[0.7],0.1)
        rospy.sleep(1)
        # action for leg going front for kicking
        self.set_joint_angles_with_speed(['LKneePitch'],[-0.08],0.7)
        rospy.sleep(2)
        # action for leg going back to home position
        self.set_joint_angles_with_speed(['LKneePitch'],[0.3],0.1)
        rospy.sleep(2)

    def central_execute(self):
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)

        #this two services are for the goalkeeper state and the execute of the action
        s = rospy.Service('executeAction', reward_msg, self.handle_server)
        s_goal = rospy.Service('getGoalKeeperState', goalkeeperstate_msg, self.handle_goalserver)
        self.stiffPub = rospy.Publisher("joint_stiffness", JointState, queue_size=10)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10)

        #self.set_stiffness(False) # always check that your robot is in a stable position before disabling the stiffness!!

        rate = rospy.Rate(10) # sets the sleep time to 10ms

        while not rospy.is_shutdown():

            #self.set_stiffness(self.stiffness)
            rate.sleep()
        #self.set_stiffness(False)
    # rospy.spin() just blocks the code from exiting, if you need to do any periodic tasks use the above loop
    # each Subscriber is handled in its own thread
    #rospy.spin()

def main():
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()


if __name__=='__main__':
    main()
