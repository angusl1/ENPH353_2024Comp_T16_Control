#!/usr/bin/env python3

# Copied code for node from http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty


class drive_forward:

    def __init__(self):
        # we want to subscribe to the image that is published automatically by the camera
        # then we want to publish the velocity which is automatically heard by the robot
        # self.image_pub = rospy.Publisher("image_topic_2", Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
        self.timer_sub = rospy.Subscriber("/clock", Clock)
        self.publish = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
        self.comp_pub = rospy.Publisher("/score_tracker", String, queue_size = 1)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # cv2.imshow('image', cv_image)
        cv2.waitKey(3) # wait 3 ms

        # convert image to hsv
        hsv_cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # defining blue mask threshold to detect clueboards
        lower_blue = np.array([115, 50, 10]) 
        upper_blue = np.array([125, 255, 255]) 

        blue_mask = cv2.inRange(hsv_cv_image, lower_blue, upper_blue)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # filtering clueboard by area (# of pixels contained within contour)
        min_area = 5000
        max_area = 30000

        potential_clueboards = []
        for contour in contours: 
            if min_area < cv2.contourArea(contour) < max_area:
                potential_clueboards.append(contour)

        if potential_clueboards:
            # avoid mistaking trees for signboard by getting biggest contour (since trees are usually smaller)
            largest_signboard = max(potential_clueboards, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_signboard)

            cropped_image = cv_image[y:y+h, x:x+w]
            cv2.imshow('Cropped Image', cropped_image)

        contour_image = cv_image.copy()

        # draw all contours in green
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

        # draw potential signboard contours in red
        cv2.drawContours(contour_image, potential_clueboards, -1, (0, 0, 255), 2)

        cv2.imshow('Contours', contour_image)

    def spawn_position(self, position):

        msg = ModelState()
        msg.model_name = 'R1'

        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.x = position[3]
        msg.pose.orientation.y = position[4]
        msg.pose.orientation.z = position[5]
        msg.pose.orientation.w = position[6]

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(msg)

        except rospy.ServiceException:
            print ("Service call failed")

    def timetrial(self):
        # Gets the velocity message from the determineVelocity function
        self.comp_pub.publish("kappa,chungus,0,ZANIEL")
        
        rospy.sleep(1)

        v = self.startVelocity()
        self.publish.publish(v)
        
        rospy.sleep(5)
        
        v = self.stopVelocity()
        self.publish.publish(v)

        self.comp_pub.publish("kappa, chungus, -1, DONE")

    # determineVelocity function calculate the velocity for the robot based
    # on the position of the line in the image.   
    def startVelocity(self):
    
        velocity = Twist()
    
        velocity.linear.x = 0
        velocity.angular.z = 0
    
        return velocity
    
    def stopVelocity(self):

        velocity = Twist()
        velocity.linear.x = 0
        velocity.angular.z = 0

        return velocity

# the main function is what is run
# calls on the image_converter class and initializes a node
def main(args):
    ic = drive_forward()
    rospy.init_node('drive_forward', anonymous=True)
    ic.timetrial()

    try:
        rospy.spin()  # spin() keeps python from exiting until the node is stopped
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
