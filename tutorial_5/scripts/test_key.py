#!/usr/bin/env python

#Group 5
# Laurie Dubois
# Bjoern Doeschl
# Gonzalo Olguin
# Dawen Zhou
# Volkan Arsan
#import keyboard
from sklearn import tree
import matplotlib.pyplot as plt
#import graphviz
import numpy as np
import pandas as pd


def image_cb(self, data):
    bridge_instance = CvBridge()
    try:
        image = bridge_instance.imgmsg_to_cv2(data, "bgr8")
        image = cv2.resize(cv_image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)

        arucoDict = cv2.aruco.Dictionary_get(cv.aruco.DICT_5X5_100)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
        aruco.drawDetectedMarkers(image, corners, ids)

    except CvBridgeError as e:
        rospy.logerr(e)

    winname = "display"
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, 400, 300)
    if len(corners) > 0:
        # 展平 ArUCo ID 列表
        ids = ids.flatten()
        # 循环检测到的 ArUCo 标记
        for (markerCorner, markerID) in zip(corners, ids):
            # 提取始终按​​以下顺序返回的标记：
            # TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # 将每个 (x, y) 坐标对转换为整数
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            # 绘制ArUCo检测的边界框
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            # 计算并绘制 ArUCo 标记的中心 (x, y) 坐标
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            # 在图像上绘制 ArUco 标记 ID
            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)
            print("[INFO] ArUco marker ID: {}".format(markerID))
            cv2.imshow(winname, image)
            cv2.waitKey(1)








def main():
    x_rew = []
    if x_rew != []:
        print(True)

if __name__=='__main__':
    main()



#reward = int(raw_input("Enter reward:"))
#print("Reward is "+ str(reward))
# while True:  # making a loop

#     if keyboard.is_pressed('up arrow'):  # if key 'q' is pressed
#         print('Reward = 10')
#         break
#     elif keyboard.is_pressed('down arrow'):  # if key 'q' is pressed
#         print('Reward = -1')
#         break
#     elif keyboard.is_pressed('right arrow'):  # if key 'q' is pressed
#         print('Reward = 0')
#         break
