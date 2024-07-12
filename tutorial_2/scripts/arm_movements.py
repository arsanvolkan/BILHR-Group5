#!/usr/bin/env python

#Group 5
# Laurie Dubois
# Bjoern Doeschl
# Gonzalo Olguin 
# Dawen Zhou
# Volkan Arsan 

from curses import KEY_RESIZE
import rospy
import roslib
from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np


class Central:

    def __init__(self):
        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
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
    # Button 1 calls the left_arm_home function putting the left arm in home position
    # Button 2 calls the left_arm_movement function to move the left arm in a repetitive motion
    # Button 3 calls the move_both_arms function to move both arms simultaneously
    def touch_cb(self,data):
        rospy.loginfo("touch button: "+str(data.button)+" state: "+str(data.state))
        
        if data.button == 1 and data.state == 1:
            rospy.loginfo("HOMING left arm...")
            self.left_arm_home()
            self.set_stiffness(False)

        elif data.button == 2 and data.state==1:
            rospy.loginfo("MOVING left arm...")
            # home first        
            self.left_arm_home()
            for i in range(3): # repeat the movement 3 times
                self.left_arm_movement() 
              
            self.left_arm_home()
            rospy.sleep(2)
            self.set_stiffness(False)
            

        elif data.button == 3 and data.state == 1:
            rospy.loginfo("Moving both arms")
            self.move_both_arms() 

            rospy.sleep(3)
            self.set_stiffness(False)

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
            if len(keypoint) != 0:
                print("keypoint coordinate x:"+ str(keypoint[0].pt[0]))
                print("keypoint coordinate y:"+str(keypoint[0].pt[1]))
            im_with_keypoints = cv2.drawKeypoints(cv_image, keypoint, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        except CvBridgeError as e:
            rospy.logerr(e)
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(3) # a small wait time is needed for the image to be displayed correctly

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

    # Unused but tried to check if the motion is complete by checking if they have no velocity.
    def check_motion(self,joint_num):
        check_int=1
        for i in joint_num:
            if (self.joint_velocities[i]!=0):
                check_int=0
                return check_int
        return check_int
              

    # This function calls the set_joint_angles()   and closes the right hand and moves the left arm to a home position
    def left_arm_home(self):

        self.set_stiffness(True)
        self.set_joint_angles(["RHand"], [0])  
        self.set_joint_angles(["LShoulderPitch","LShoulderRoll","LElbowYaw","LElbowRoll","LWristYaw"],[1.65,0.06,-0.847,-0.05,-0.5])
        rospy.sleep(3)
    

    #This function let the robot wave with its hand in a way.

    def left_arm_movement(self):
       
        #arm up 
        self.set_stiffness(True)
        self.set_joint_angles(["LShoulderPitch","LShoulderRoll","LElbowYaw","LElbowRoll","LWristYaw"],[0.075,0.059,0.061,-0.094,-1.66])       
       
        rospy.sleep(2.5)
        
        #turn wrist
        self.set_stiffness(True)
        self.set_joint_angles(["LShoulderPitch","LShoulderRoll","LElbowYaw","LElbowRoll","LWristYaw"],[0.075,0.059,0.061,-0.094,1])       
        rospy.sleep(0.8)
        self.set_joint_angles(["LShoulderPitch","LShoulderRoll","LElbowYaw","LElbowRoll","LWristYaw"],[0.075,0.059,0.061,-0.094,-1.66]) 
        rospy.sleep(0.8)    
        self.set_stiffness(True)

    #This funtion moves both arms with the same motion.
    #Therefore the joint_angles were negated for the right hand.
    def move_both_arms(self):
        self.set_stiffness(True)

        self.set_joint_angles(["RHand"], [0])        
        self.set_joint_angles(["LShoulderPitch","LShoulderRoll","LElbowYaw","LElbowRoll","LWristYaw",
        "RShoulderPitch","RShoulderRoll","RElbowYaw","RElbowRoll","RWristYaw" ],[1.65,0.06,-0.847,-0.05,-0.5, 1.65,0.06,0.847,0.05,0.5])
        rospy.sleep(2.5)
    

        #turn wrist
        self.set_stiffness(True)
        self.set_joint_angles(["LShoulderPitch","LShoulderRoll","LElbowYaw","LElbowRoll","LWristYaw",
        "RShoulderPitch","RShoulderRoll","RElbowYaw","RElbowRoll","RWristYaw" ],[0.075,0.059,0.061,-0.094,1, 0.075,0.059,-0.061,0.094,-1])       
        rospy.sleep(2)
        self.set_joint_angles(["LShoulderPitch","LShoulderRoll","LElbowYaw","LElbowRoll","LWristYaw",
        "RShoulderPitch","RShoulderRoll","RElbowYaw","RElbowRoll","RWristYaw" ],[0.075,0.059,0.061,-0.094,-1.66, 0.075,0.059,-0.061,0.094,1.66]) 
        rospy.sleep(2)

        #home
        self.set_joint_angles(["LShoulderPitch","LShoulderRoll","LElbowYaw","LElbowRoll","LWristYaw",
        "RShoulderPitch","RShoulderRoll","RElbowYaw","RElbowRoll","RWristYaw" ],[1.65,0.06,-0.847,-0.05,-0.5, 1.65,0.06,0.847,0.05,0.5])
        rospy.sleep(2.5)

    def central_execute(self):
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10)

        #self.set_stiffness(False) # always check that your robot is in a stable position before disabling the stiffness!!

        rate = rospy.Rate(10) # sets the sleep time to 10ms

        while not rospy.is_shutdown():

            #self.set_stiffness(self.stiffness)
            rate.sleep()

    # rospy.spin() just blocks the code from exiting, if you need to do any periodic tasks use the above loop
    # each Subscriber is handled in its own thread
    #rospy.spin()



if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
