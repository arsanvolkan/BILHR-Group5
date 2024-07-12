#!/usr/bin/env python

#Group 5
# Laurie Dubois
# Bjoern Doeschl
# Gonzalo Olguin
# Dawen Zhou
# Volkan Arsan

from curses import KEY_RESIZE
import rospy

from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import json
import csv
from tutorial_3.srv import *


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
        print("Init")

        pass


    def key_cb(self,data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def joints_cb(self,data):
        # store current joint information in class variables
        self.joint_names = data.name
        self.joint_angles = data.position
        self.joint_velocities = data.velocity

        pass

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
            
            self.move_to_blob()
            pass

        elif data.button == 2 and data.state==1:
            self.set_stiffness(False)
            pass


        elif data.button == 3 and data.state == 1:
            self.set_stiffness(True)
            self.setToHomePosition()
            #self.set_stiffness_Task()
            #self.setToHomePosition()
            #self.start_get_data(150)




    def move_to_blob(self):
        x = self.blob_pos_x
        y = self.blob_pos_y

        if x == 0 and y == 0:
            print("No blob detected.")
            return
        else:
            rospy.wait_for_service('predict_joints')
            #print(x)
            #print(y)
            try:

                predict_joints_client = rospy.ServiceProxy('predict_joints', predictJoints)
                #calls the network service to do a forward step an it returns the angles.
                resp = predict_joints_client(x, y)
                #print(resp.angle1)
                #print(resp.angle2)
                # set angle on resp.angle1 and resp.angle2
                self.set_joint_angles(["LShoulderPitch","LShoulderRoll"],[resp.angle1, resp.angle2])
                
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)

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
            cv_image = bridge_instance.imgmsg_to_cv2(data,"bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            lower_red = np.array([0,150,20])    #([0,50,220])
            upper_red = np.array([5,255,255])    #([10,255,255])
            mask0 = cv2.inRange(hsv, lower_red, upper_red)

            lower_red = np.array([175,150,20])
            upper_red = np.array([180,255,255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)

            mask = mask0 + mask1
            output_hsv = hsv.copy()
            output_hsv[np.where(mask==0)] = 0
            output_hsv[np.where(mask!=0)] = 255
            kernel=np.ones((5,5),np.uint8)
            img_erosion=cv2.erode(output_hsv,kernel,iterations=1)
            output_img=cv2.dilate(img_erosion,kernel,iterations=1)

            params = cv2.SimpleBlobDetector_Params()
            # Thresholds for grayscale image
            params.minThreshold = 10
            params.maxThreshold = 200

            #params set to false
            params.filterByInertia = False
            params.filterByConvexity = False

            #to let it detect the white areas not black
            params.filterByColor = True
            params.blobColor = 255

            #to ignore little blobs and to allow big blobs
            params.filterByArea=True
            params.minArea = 1500
            params.maxArea = 1000000

            params.filterByCircularity = True
            params.minCircularity = 0.1

            detector=cv2.SimpleBlobDetector_create(params)

            keypoint=detector.detect(output_img)
            # if len(keypoint) != 0:
            #     print("keypoint coordinate x:"+ str(keypoint[0].pt[0]))
            #     print("keypoint coordinate y:"+str(keypoint[0].pt[1]))
            if len(keypoint) != 0:
                self.blob_pos_x = keypoint[0].pt[0]
                self.blob_pos_y = keypoint[0].pt[1]
                self.move_to_blob()
                

            im_with_keypoints = cv2.drawKeypoints(cv_image, keypoint, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        except CvBridgeError as e:
            rospy.logerr(e)
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(3) # a small wait time is needed for the image to be displayed correctly

    def setToHomePosition(self):
        """Sets the roboter to the home position. Camera tilted left and hand closed"""
        self.set_joint_angles(["HeadYaw","HeadPitch","LHand"],[0.3622,-0.105,0])
        rospy.sleep(2)


    def start_get_data(self,number_of_samples):
        """For data acquisition. Every time enter is pressed the actual blob coordinates are stored
        and the respective roll pitch values. And then stored into a .csv file"""
        size = 0
        store = []
        while size < number_of_samples:
            raw_input("Press Enter to record data...")
            blob_x = self.blob_pos_x
            blob_y = self.blob_pos_y
            if blob_x == 0 and blob_y == 0:
                print("INFO BLOBS COORD ARE ZERO")
            Lshoulder_pitch = self.joint_angles[2]
            Lschoulder_roll = self.joint_angles[3]
            store.append(Lshoulder_pitch)
            store.append(Lschoulder_roll)
            store.append(blob_x)
            store.append(blob_y)
            size = (len(store))/4
            print(size)

        #self.write_json_file(store)
        self.write_csv_file(store)

    def write_csv_file(self,data):
        """Write the data into a csv file."""
        header=["id","Lshoulder_pitch","Lshoulder_roll","blob_x","blob_y"]
        with open('./data/raw_data_new.csv', 'w') as outfile:
            writer=csv.writer(outfile)
            writer.writerow(header)
            for i in range(0, len(data), 4):
                temp_row=[int(i / 4),data[0 + i],data[1 + i],data[2 + i],data[3 + i]]
                writer.writerow(temp_row)
        print("done writing file, all {} data have been written".format(len(data) / 4))

    def set_stiffness_Task(self):
        """Set the stiffness for the data acquisition. So the shoulder can be moved"""
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

    # This function calls the set_joint_angles()   and closes the right hand and moves the left arm to a home position
    def left_arm_home(self):

        self.set_stiffness(True)
        self.set_joint_angles(["RHand"], [0])
        self.set_joint_angles(["LShoulderPitch","LShoulderRoll","LElbowYaw","LElbowRoll","LWristYaw"],[1.65,0.06,-0.847,-0.05,-0.5])
        rospy.sleep(3)


    def central_execute(self):
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)
        self.stiffPub = rospy.Publisher("joint_stiffness", JointState, queue_size=10)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10)

        #self.set_stiffness(False) # always check that your robot is in a stable position before disabling the stiffness!!

        rate = rospy.Rate(10) # sets the sleep time to 10ms

        while not rospy.is_shutdown():

            #self.set_stiffness(self.stiffness)
            rate.sleep()

    # rospy.spin() just blocks the code from exiting, if you need to do any periodic tasks use the above loop
    # each Subscriber is handled in its own thread
    #rospy.spin()

def main():
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()


if __name__=='__main__':
    main()
