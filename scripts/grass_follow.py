#!/usr/bin/env python3
from __future__ import print_function

import roslib
import sys
import rospy
import cv2 
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import numpy as np

MAX_TIME = 1000

class state_manager:
  def __init__(self):
    self.bridge = CvBridge()

    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
    self.timer_sub = rospy.Subscriber("/clock", Clock)
    self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
    self.comp_pub = rospy.Publisher("/score_tracker", String, queue_size = 1)

    self.crosswalk = True # Flag for detecting crosswalk
    self.past_error = 0
    
    rospy.sleep(1)
    self.comp_pub.publish("kappa,chungus,0,ZANIEL")
    self.start_time = rospy.get_time()

  def GrassFollowing(self,frame):

    kernel = np.ones((5,5),np.uint8)

    twist = Twist()
    twist.linear.x = 0.3

    height, width, _ = frame.shape
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    max_height = 360 # 270
    bot_height = 60
    roi_frame = hsv_frame[height-max_height:height-bot_height, 0:width]
    centroid_x = width / 2
    centroid_y = height / 2

    # Define the lower and upper bounds for sides of the path
    lower = np.array([25, 30, 180])   
    upper = np.array([80, 70, 215]) 

    # Create a mask for path sides and remove noise
    mask = cv2.inRange(roi_frame, lower, upper)
    # mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 3)
    mask = cv2.erode(mask, kernel, iterations = 3)
    mask = cv2.dilate(mask, kernel, iterations = 5)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) >= 2:
        # Find the contour with the largest area (assuming it's the path)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Remove the largest contour to find the second largest contour (assume other side of path)
        max_contour1 = contours[0]
        max_contour2 = contours[1]

        # Find the centroid of the largest contour
        M1 = cv2.moments(max_contour1)
        M2 = cv2.moments(max_contour2)

        if M1['m00'] != 0 and M2['m00'] != 0:
            cx1 = int(M1['m10'] / M1['m00'])
            cy1 = int(M1['m01'] / M1['m00'])
            cx2 = int(M2['m10'] / M2['m00'])
            cy2 = int(M2['m01'] / M2['m00'])

            centroid_x = int(np.average([cx1, cx2]))
            centroid_y = int(np.average([cy1, cy2]))
            # Draw a red circle at the centroid position
            cv2.circle(mask, (centroid_x, centroid_y), radius=20, color=(0, 0, 255), thickness=-1)

    cv2.imshow("Mask window", mask)
    cv2.waitKey(3)

    error = int(width/2) - centroid_x

    #PID well i guess only P
    P = 0.02
    I = 0.01
    min_error = 25

    if np.abs(error) > min_error:
      twist.angular.z = P * error - I * (error - self.past_error)
    else: 
      twist.angular.z = 0

    self.past_error = error
    
    return twist

  def callback(self,data):
      try:
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      except CvBridgeError as e:
        print(e)

  def start(self):
      
      # Set time limit for run
      while (not rospy.is_shutdown()) and (rospy.get_time() - self.start_time < MAX_TIME):

        # Show camera feed
        cv2.imshow("Image window", self.cv_image)
        cv2.waitKey(3)

        # Line following
        try:
          self.vel_pub.publish(self.GrassFollowing(self.cv_image))
        except CvBridgeError as e:
          print(e)

        # Check if the crosswalk has been detected, only runs if crosswalk has not been detected yet
        if self.crosswalk:

          # Initiate a sequence if the crosswalk is detected
          self.detect_crosswalk(self.cv_image)


      # End message
      self.comp_pub.publish("kappa, chungus, -1, DONE")
      
      # Stop the robot
      self.vel_pub.publish(self.stop_robot())

  # Detect the crosswalk contour
  def detect_crosswalk(self, frame):
    height, width, _ = frame.shape
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    max_height = 300
    bot_height = 0
    roi_frame = hsv_frame[height-max_height:height-bot_height, 0:width]
    cx = 0

    # Define the lower and upper bounds for crosswalk line
    lower_red = np.array([0, 50, 150])
    upper_red = np.array([15, 255, 255])

    # Create a mask for the crosswalk line
    red_mask = cv2.inRange(roi_frame, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the area of the largest contour (assuming it's the crosswalk line)
        max_contour_area = cv2.contourArea(max(contours, key=cv2.contourArea))
        
        # Check if area of max contour is large enough
        if max_contour_area > 30000:
           self.crosswalk = False
           print(max_contour_area)
           try:
            self.vel_pub.publish(self.stop_robot())
           except CvBridgeError as e:
            print(e)
           rospy.sleep(2)

    #cv2.imshow("Mask window", red_mask)
    #cv2.waitKey(3)
    pass

  # Return a twist object which renders the robot stationary
  def stop_robot(self):
      
      twist = Twist()
      twist.angular.z = 0
      twist.linear.x = 0
      
      return twist     
  
  def reset_position(self):

      msg = ModelState()
      msg.model_name = 'R1'

      msg.pose.position.x = 0.5
      msg.pose.position.y = -0.5
      msg.pose.position.z = 0.2
      msg.pose.orientation.x = 0
      msg.pose.orientation.y = 0
      msg.pose.orientation.z = 1
      msg.pose.orientation.w = 1

      # msg.pose.position.x = 5.5
      # msg.pose.position.y = 2.5
      # msg.pose.position.z = 0.2
      # msg.pose.orientation.x = 0
      # msg.pose.orientation.y = 0
      # msg.pose.orientation.z = -1
      # msg.pose.orientation.w = 1

      try:
        self.vel_pub.publish(self.stop_robot())
      except CvBridgeError as e:
        print(e)

      rospy.wait_for_service('/gazebo/set_model_state')
      try:
          set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
          resp = set_state(msg)
      except rospy.ServiceException:
          print ("Service call failed")
      
      



def main(args):
  rospy.init_node('image_converter', anonymous=True)
  rob = state_manager()
  rob.reset_position()
  rob.start()
  # rospy.sleep(5)
  print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
