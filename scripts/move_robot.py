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

    self.get_image = False # Flag for getting clueboard image
    self.clueboard = True # Flag for seen clueboard
    self.crosswalk = True # Flag for detecting crosswalk

    self.past_error = 0 # For I in line following
    
    rospy.sleep(1)
    self.comp_pub.publish("kappa,chungus,0,ZANIEL")
    self.start_time = rospy.get_time()
    self.clueboard_time = 0

  def RoadFollowing(self,frame):

    twist = Twist()
    twist.linear.x = 0.4

    height, width, _ = frame.shape
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    max_height = 270
    bot_height = 0
    roi_frame = hsv_frame[height-max_height:height-bot_height, 0:width]
    cx = 0

    # Define the lower and upper bounds for gray
    lower_gray = np.array([0, 0, 80])
    upper_gray = np.array([180, 10, 90])

    # Create a mask for the gray path
    gray_mask = cv2.inRange(roi_frame, lower_gray, upper_gray)

    # Find contours in the mask
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the contour with the largest area (assuming it's the path)
        max_contour = max(contours, key=cv2.contourArea)

        # Find the centroid of the largest contour
        M = cv2.moments(max_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Draw a red circle at the centroid position
            cv2.circle(gray_mask, (cx, cy), radius=20, color=(0, 0, 255), thickness=-1)

    #cv2.imshow("Mask window", gray_mask)
    #cv2.waitKey(3)

    error = int(width/2) - cx

    #PID well i guess only P
    P = 0.020
    I = 0.010
    min_error = 25

    if np.abs(error) > min_error:
      pass
      twist.angular.z = P * error - I * (error - self.past_error)
    else: 
      twist.angular.z = 0

    self.past_error = error
    
    return twist
  
  def get_clueboard(self, frame):
    height, width, _ = frame.shape
    area = height * width

    if self.get_image:
      if (rospy.get_time() - self.clueboard_time) > 1.0:
        self.get_image = False
      return  
    
    if area > 35000:
      # save the first time you see a clueboard
      self.clueboard_time = rospy.Time.now().to_sec()

      cv2.imwrite('clueboard.jpg', frame)
      clueboard_image = cv2.imread('clueboard.jpg')
      height, width, _ = clueboard_image.shape

      if clueboard_image is not None:
        # cv2.imshow('Clueboard Image', clueboard_image)

        # convert to grayscale
        gray_image = cv2.cvtColor(clueboard_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)  
        binary_inverted = cv2.bitwise_not(binary)

        # remove white border
        contours, _ = cv2.findContours(binary_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
          # mask to remove image border
          largest_contour = max(contours, key=cv2.contourArea)
          mask = np.zeros_like(gray_image)
          cv2.drawContours(mask, [largest_contour], -1, (255), cv2.FILLED)
          removed_border_image = cv2.bitwise_and(clueboard_image, clueboard_image, mask=mask)
          height,width,_ = removed_border_image.shape

          # bounding boxes around words
          letters_gray = cv2.cvtColor(removed_border_image, cv2.COLOR_BGR2GRAY)

          letter_binary = cv2.adaptiveThreshold(letters_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

          letter_contours, _ = cv2.findContours(letter_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

          letter_image = removed_border_image.copy()
          cv2.drawContours(letter_image, letter_contours, -1, (0, 255, 0), 1)

          cv2.imshow('Bounding Boxes around Words', letter_image)

          for contour in letter_contours:
             x, y, w, h = cv2.boundingRect(contour)
             removed_border_image = cv2.rectangle(removed_border_image.copy(), (x,y), (x+w, y+h), (0, 255, 0), 2)

          cv2.imshow('Bounding Rectangle around Words', removed_border_image)
      
        # erosion

        # cv2.imshow('Result', removed_border_image)
        # kernel = np.ones((3, 3), np.uint8)

        # thinner_image = cv2.erode(binary_image, kernel, iterations=1)
        # contours, _ = cv2.findContours(thinner_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contour_image = cv2.drawContours(binary_image.copy(), contours, -1, (0, 255, 0), 2)
        # cv2.imshow('Clueboard Contours', contour_image)

        # cv2.imshow('Eroded Image', thinner_image)

      self.get_image = True

  def find_clueboard(self, frame):
    # convert image to hsv
    hsv_cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)

    # defining blue mask threshold to detect clueboards
    lower_blue = np.array([118, 50, 20]) 
    upper_blue = np.array([122, 255, 255]) 

    blue_mask = cv2.inRange(hsv_cv_image, lower_blue, upper_blue)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filtering possible clueboards by area (# of pixels contained within contour)
    min_area = 5000
    max_area = 50000

    potential_clueboards = []
    for contour in contours: 
        if min_area < cv2.contourArea(contour) < max_area:
            potential_clueboards.append(contour)

    if potential_clueboards:
        # avoid mistaking tree outlines for clueboard by getting biggest contour
        largest_clueboard = max(potential_clueboards, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_clueboard)

        cropped_image = self.cv_image[y:y+h, x:x+w]

        # cv2.imshow('Cropped Image', cropped_image)

        # polygon approximation of clueboard
        epsilon = 0.05 * cv2.arcLength(largest_clueboard, True)
        approximate_contour = cv2.approxPolyDP(largest_clueboard, epsilon, True)

        if len(approximate_contour) == 4:
            # corners of the clueboard
            corners = np.array([point[0] for point in approximate_contour])
                
            col, row, _ = cropped_image.shape

            # calculate average x and y coordinates of the corners
            avg_x = np.mean(corners[:, 0]) 
            avg_y = np.mean(corners[:, 1]) 

            top_left = (0, 0)
            bottom_left = (0, 0)
            top_right = (0, 0)
            bottom_right = (0, 0)

            # Iterate through corners
            for corner in corners:
                x, y = corner
                
                # assign points as top/bottom right/left by comparing to avg_x,y
                if x < avg_x:
                    if y > avg_y:
                        top_left = corner 
                    else:
                        bottom_left = corner
                else:
                    if y > avg_y:
                        top_right = corner 
                    else:
                        bottom_right = corner

            pts1 = np.float32([bottom_left, bottom_right, top_left, top_right])
            pts2 = np.float32([[0, 0], [row, 0], [0, col], [row, col]])
            
            # perspective transform matrix and transform image
            pt_matrix = cv2.getPerspectiveTransform(pts1, pts2)
            pt_image = cv2.warpPerspective(self.cv_image, pt_matrix, (row, col))

            # cv2.imshow('Transformed Image', pt_image)

            # second blue mask 
            hsv_pt_image = cv2.cvtColor(pt_image, cv2.COLOR_BGR2HSV)
            pt_blue_mask = cv2.inRange(hsv_pt_image, lower_blue, upper_blue)

            blue_mask_resized = cv2.resize(pt_blue_mask, (pt_image.shape[1], pt_image.shape[0]))

            # turn transformed image black and white
            bw_image = np.zeros_like(pt_image)
            bw_image[np.where(blue_mask_resized == 255)] = 255

            cv2.imshow('Black & White Image', bw_image)

            self.get_clueboard(bw_image)

    contour_image = self.cv_image.copy()

    # draw all contours in green
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # draw potential signboard contours in red
    cv2.drawContours(contour_image, potential_clueboards, -1, (0, 0, 255), 2)

    cv2.imshow('Contours', contour_image)     

  def callback(self,data):
      try:
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      except CvBridgeError as e:
        print(e)

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
           rospy.sleep(5)

    # cv2.imshow("Mask window", red_mask)
    cv2.waitKey(3)
    pass

  # Return a twist object which renders the robot stationary
  def stop_robot(self):
      
      twist = Twist()
      twist.angular.z = 0
      twist.linear.x = 0
      
      return twist

  # Return a twist object which makes the robot go forward
  def forward_robot(self):

    twist = Twist()
    twist.angular.z = 0
    twist.linear.x = 0.5

  def reset_position(self):

      msg = ModelState()
      msg.model_name = 'R1'

      msg.pose.position.x = 5.5
      msg.pose.position.y = 2.5
      msg.pose.position.z = 0.2
      msg.pose.orientation.x = 0
      msg.pose.orientation.y = 0
      msg.pose.orientation.z = -1
      msg.pose.orientation.w = 1

      rospy.wait_for_service('/gazebo/set_model_state')
      try:
          set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
          resp = set_state(msg)

      except rospy.ServiceException:
          print ("Service call failed")

  # Start function, considers states
  def start(self):
      
      # Set time limit for run
      while (not rospy.is_shutdown()) and (rospy.get_time() - self.start_time < MAX_TIME):

        # Show camera feed
        # cv2.imshow("Image window", self.cv_image)
        cv2.waitKey(3)

        # Line following
        try:
          self.vel_pub.publish(self.RoadFollowing(self.cv_image))
          self.find_clueboard(self.cv_image)
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

def main(args):
  rospy.init_node('image_converter', anonymous=True)
  rob = state_manager()
  rob.start()
  # rospy.sleep(5)
  rob.reset_position()
  print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
