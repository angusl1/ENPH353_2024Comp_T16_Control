#!/usr/bin/env python3
from __future__ import print_function

import roslib
import sys
import rospy
import cv2 
import os
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from PIL import Image as pil

import clue_model as cmodel
import numpy as np

MAX_TIME = 1000

class state_manager:
  def __init__(self):
    self.bridge = CvBridge()
    self.kernel = np.ones((5,5),np.uint8)
    self.id = 0

    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
    self.timer_sub = rospy.Subscriber("/clock", Clock)
    self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
    self.comp_pub = rospy.Publisher("/score_tracker", String, queue_size = 1)

    # State flags
    self.get_image = False # Flag for getting clueboard image
    self.clueboard_count = 0 # Count for seen clueboard
    self.crosswalk = True # Flag for detecting crosswalk
    self.pink_line_count = 0 # Flag for counting pink lines
    self.last_pink_time = 0 # Time at which last pink line was detected
    self.pink1_time = 0 # Time at which first pink line is read
    self.past_cb3 = False # Flag for passing clueboard 3
    self.past_cb5 = False # Flag for reading clueboard 5
    self.past_cb6 = False # Flag for reading clueboard 6
    self.yoda_found = False # Flag for detecting if Yoda has been found
    self.tunnel_flag = False # False if tunnel not passed
    self.finished_flag = False # Finished tunnel sequence
    

    self.past_error = 0 # For I in line following
    self.num_letters = 0 # number of letters in a word
    self.area_threshold = 30000

    self.prediction_model = cmodel.clue_model()
    
    rospy.sleep(1)
    self.comp_pub.publish("kappa,chungus,0,ZANIEL")
    self.start_time = rospy.get_time()
    self.clueboard_time = 0

  def RoadFollowing(self,frame):

    twist = Twist()
    twist.linear.x = 0.35

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
        
        if cv2.contourArea(max_contour) > 300000:
         twist.angular.z = 4
         return twist

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
  
  # removes white border of images
  def crop_border(self):
      clueboard_image = cv2.imread('clueboard.jpg')

      if clueboard_image is not None:

        # convert to grayscale
        gray_image = cv2.cvtColor(clueboard_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)  
        binary_inverted = cv2.bitwise_not(binary)

        # remove white border
        contours, _ = cv2.findContours(binary_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
          # mask to remove image border
          largest_contour = max(contours, key=cv2.contourArea)
          mask = np.zeros_like(gray_image)
          cv2.drawContours(mask, [largest_contour], -1, (255), cv2.FILLED)
          removed_border_image = cv2.bitwise_and(clueboard_image, clueboard_image, mask=mask) 
          return removed_border_image
        
      return clueboard_image

  # cleans up image by removing border then gets images of letters and puts them in a separate file
  def get_letters(self, frame):
    height, width, _ = frame.shape
    area = height * width
    aspect_ratio = width / height

    if self.get_image:
      if (rospy.get_time() - self.clueboard_time) > 7:
        self.get_image = False
      return  
    
    if area > self.area_threshold and aspect_ratio < 2.0 and aspect_ratio > 1.0:
      # save the first time you see a clueboard
      print("Area: ", area)
      print("Threshold:", self.area_threshold)
      self.clueboard_time = rospy.Time.now().to_sec()

      cv2.imwrite('clueboard.jpg', frame)
      clueboard_image = cv2.imread('clueboard.jpg')

      if clueboard_image is not None:
        removed_border_image = self.crop_border()
        borderless_h, borderless_w, _ = removed_border_image.shape

        # bounding boxes around words
        letters_gray = cv2.cvtColor(removed_border_image, cv2.COLOR_BGR2GRAY)

        letter_binary = cv2.adaptiveThreshold(letters_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

        letter_contours, _ = cv2.findContours(letter_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        letter_image = removed_border_image.copy()

        # filter contours by size to only get the letter contours not the outline of the word, and get rid of small blemishes
        filtered_contours = []
        image_area = borderless_h * borderless_w

        for i, lc in enumerate(letter_contours):

          contour_area = cv2.contourArea(lc)
          if contour_area > 0:
            if image_area / contour_area < 500 and contour_area < 1000:
              filtered_contours.append(lc)

        # ordering contours: top words first then bottom words
        top_word = []
        bottom_word = []

        filtered_contours = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[1])

        for lc in filtered_contours:
          _, y1, _, _ = cv2.boundingRect(lc)
          if y1 < frame.shape[0] / 2:
            top_word.append(lc)
          else:
            bottom_word.append(lc)

        top_word = sorted(top_word, key=lambda c: cv2.boundingRect(c)[0])
        bottom_word = sorted(bottom_word, key=lambda c: cv2.boundingRect(c)[0])

        sorted_letters = []
        print(borderless_w)
        print(image_area)

        for i, lc in enumerate(bottom_word):
          x, y, w, h = cv2.boundingRect(lc)

          if borderless_h / h < 5.5 or borderless_h / h > 9.0:
            bottom_word.pop(i)

        for lc in bottom_word:
          x, y, w, h = cv2.boundingRect(lc)
          width_ratio = borderless_w / w
          print(borderless_w / w)

          if width_ratio < 10 and width_ratio > 7.5:
            mid_x = x + w // 2
            roi_box1 = frame[y:y+h, x:mid_x]
            roi_box2 = frame[y:y+h, mid_x:x+w]
            sorted_letters.append(roi_box1)
            sorted_letters.append(roi_box2)

          elif width_ratio <= 7.5:
            third_x = x + w // 3
            two_third_x = x + 2 * w // 3
            roi_box11 = frame[y:y+h, x:third_x]
            roi_box12 = frame[y:y+h, third_x:two_third_x]
            roi_box13 = frame[y:y+h, two_third_x:x+w]
            sorted_letters.append(roi_box11)
            sorted_letters.append(roi_box12)
            sorted_letters.append(roi_box13)

          else:
            letter_roi = frame[y:y+h, x:x+w]
            sorted_letters.append(letter_roi)
        
        prediction = ""

        try:
          self.vel_pub.publish(self.stop_robot())
        except CvBridgeError as e:
          print(e)

        for i, letter in enumerate(sorted_letters):
          if letter.size != 0:
            letter_pil = pil.fromarray(letter)
            cv2.imwrite(os.path.join('/home/fizzer/ros_ws/src/353CompT16Controller/scripts/letters', f"letter_{i}.png"), letter)
            prediction_letter = self.prediction_model.predict(letter_pil)
            prediction = prediction + prediction_letter
            print(prediction)
          else:
            print(f"SKIP {i}")
        
        score_tuple = ("kappa", "chungus", str(self.clueboard_count+1), str(prediction))
        score_msg = ",".join(score_tuple)
        self.comp_pub.publish(score_msg)

        cv2.drawContours(letter_image, bottom_word, -1, (0, 0, 255), 1)
        cv2.imshow('Bounding Boxes around letters', letter_image)

        # area_thresholds = [25000, 18000, 16000, 20000, 8000, 16000, 20000, 30000] # Good
        area_thresholds = [25000, 25000, 20000, 25000, 20000, 20000, 20000, 30000, 30000]
        self.area_threshold = area_thresholds[self.clueboard_count+1]

        self.clueboard_count = self.clueboard_count + 1
        print("Clueboard count: " ,(self.clueboard_count))

      self.get_image = True

  def find_clueboard(self, frame):
    # convert image to hsv
    hsv_cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

        cropped_image = frame[y:y+h, x:x+w]

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
            pt_image = cv2.warpPerspective(frame, pt_matrix, (row, col))

            # mask for letters
            hsv_pt_image = cv2.cvtColor(pt_image, cv2.COLOR_BGR2HSV)

            lower_blue2 = np.array([115, 90, 30])
            upper_blue2 = np.array([120, 255, 204])

            pt_blue_mask = cv2.inRange(hsv_pt_image, lower_blue2, upper_blue2)

            blue_mask_resized = cv2.resize(pt_blue_mask, (pt_image.shape[1], pt_image.shape[0]))

            # turn transformed image black and white
            bw_image = np.zeros_like(pt_image)
            bw_image[np.where(blue_mask_resized == 255)] = 255

            self.get_letters(bw_image)

    contour_image = frame.copy()

    # draw all contours in green
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # draw potential clueboard contours in red
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
        if max_contour_area > 60000:
           print("Sidewalk Red Line Area: " ,(max_contour_area))
           try:
            self.vel_pub.publish(self.stop_robot())
           except CvBridgeError as e:
            print(e)
           while self.crosswalk:
              self.pedestrian_avoidance(self.cv_image)

    # cv2.imshow("Mask window", red_mask)
    cv2.waitKey(3)

  def pedestrian_avoidance(self, frame):
    # Define region of interest
    height, width, _ = frame.shape
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    max_height = 400
    bot_height = 100
    roi_frame = hsv_frame[height-max_height:height-bot_height, 0:width]

    # Mask the jeans
    low_mask = np.array([80, 50, 50])
    upper_mask = np.array([140, 150, 150])
    mask = cv2.inRange(roi_frame, low_mask, upper_mask)
    mask = cv2.erode(mask, self.kernel, iterations = 1)
    mask = cv2.dilate(mask, self.kernel, iterations = 3)

    #cv2.imshow("Ped Window", mask)
    #cv2.waitKey(3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
      max_contour = max(contours, key=cv2.contourArea)

      # Find the centroid of the largest contour
      M = cv2.moments(max_contour)
      if M['m00'] != 0 and cv2.contourArea(max_contour) > 400:
          cx = int(M['m10'] / M['m00'])
          cy = int(M['m01'] / M['m00'])

          # Draw a red circle at the centroid position
          cv2.circle(mask, (cx, cy), radius=20, color=(0, 0, 255), thickness=-1)
          
          if cx > 440 and cx < 840:
            print('ped seen in middle moving soon')
            rospy.sleep(0.45)
            try:
              self.vel_pub.publish(self.forward_robot())
            except CvBridgeError as e:
              print(e)
            rospy.sleep(0.5)
            self.crosswalk = False

  # Activate after third sign
  # If the truck is close stop the robot
  # Follows contours with left bias when contour is large
  def roundabout_follow(self, frame):

    kernel = np.ones((5,5),np.uint8)

    twist = Twist()
    twist.linear.x = 0.5

    height, width, _ = frame.shape
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    max_height = 300
    bot_height = 0
    roi_frame = hsv_frame[height-max_height:height-bot_height, 0:width]
    cx1 = cx2 =  cy1 = cy2 = 0

    # Define the lower and upper bounds for gray
    lower_gray = np.array([0, 0, 80])
    upper_gray = np.array([180, 10, 90])

    # Create a mask for the gray path
    mask = cv2.inRange(roi_frame, lower_gray, upper_gray)
    #mask = cv2.erode(mask, kernel, iterations = 3)
    mask = cv2.dilate(mask, kernel, iterations = 4)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
    # Find the contour with the largest area (assuming it's the path)
      max_contour = max(contours, key=cv2.contourArea)
      # print(cv2.contourArea(max_contour))

      if cv2.contourArea(max_contour) > 300000:
         twist.angular.z = 5
         return twist

    # Find the centroid of the largest contour
      M = cv2.moments(max_contour)
      if M['m00'] != 0:
          cx1 = int(M['m10'] / M['m00'])
          cy1 = int(M['m01'] / M['m00'])

          # Draw a red circle at the centroid position
          cv2.circle(mask, (cx1, cy1), radius=20, color=(0, 0, 255), thickness=-1)

    #cv2.imshow("Mask window", mask)
    #cv2.waitKey(3)

    error = int(width/2) - cx1

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
  
  # Stops the robot if the truck is close
  # Finds contour of truck and stops if the contour is large
  def truck_detect(self,frame):

    # Make sure we are past the 3rd clueboard
    if self.past_cb3 == False:
      cb3_time = rospy.get_time()
      wait_timer = 3
      while rospy.get_time() - cb3_time < wait_timer:
        try:
          self.vel_pub.publish(self.RoadFollowing(self.cv_image))
        except CvBridgeError as e:
          print(e)
      self.past_cb3 = True

    kernel = np.ones((5,5),np.uint8)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for truck
    lower = np.array([0, 0, 120])   
    upper = np.array([20, 20, 205]) 

    # Create a mask for path sides and remove noise
    mask = cv2.inRange(hsv_frame, lower, upper)
    mask = cv2.erode(mask, kernel, iterations = 2)
    mask = cv2.dilate(mask, kernel, iterations = 3)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
      
        # Stop the robot if the truck is too close
        if cv2.contourArea(max_contour) > 4000:
          # cv2.imshow("Truck window", mask)
           cv2.waitKey(3)
           return self.stop_robot()
        else:
           return self.roundabout_follow(frame) 
           
    return self.roundabout_follow(frame) # Return a road following twist if the truck contour is not found
        
  # Looks for the pink line and increments counter
  def detect_pink(self, frame):
    height, width, _ = frame.shape
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    max_height = height
    bot_height = 0
    roi_frame = hsv_frame[height-max_height:height-bot_height, 0:width]
    cx = 0

    # Define the lower and upper bounds for pink line
    lower_red = np.array([150, 180, 128])
    upper_red = np.array([170, 255, 255])

    # Create a mask for the crosswalk line
    red_mask = cv2.inRange(roi_frame, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the area of the largest contour (assuming it's the crosswalk line)
        max_contour_area = cv2.contourArea(max(contours, key=cv2.contourArea))
        
        # Check if area of max contour is large enough
        if max_contour_area > 20000:
           print("Pink Line Detected.\nArea:" ,(max_contour_area))
           self.pink_line_count = self.pink_line_count + 1
           if self.pink_line_count == 1 and self.pink1_time == 0:
              self.pink1_time = rospy.get_time()
           print("Pink line count:", self.pink_line_count)
           self.last_pink_time = rospy.get_time()
        elif max_contour_area > 400:
           self.find_clueboard(frame)
           self.follow_pink(contours)              

  # Follows the pink line
  def follow_pink(self, contours):
    
    # Find the contour with the largest area (assuming it's the path)
    max_contour = max(contours, key=cv2.contourArea)

    # Find the centroid of the largest contour
    M = cv2.moments(max_contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    #cv2.imshow("Mask window", gray_mask)
    #cv2.waitKey(3)

    error = 640 - cx

    #PID well i guess only P
    P = 0.020
    I = 0.010
    min_error = 25

    twist = Twist()
    twist.linear.x = 0.4

    if np.abs(error) > min_error:
      pass
      twist.angular.z = P * error - I * (error - self.past_error)
    else: 
      twist.angular.z = 0

    self.past_error = error
    
    try:
      self.vel_pub.publish(twist)
    except CvBridgeError as e:
      print(e)

    self.detect_pink(self.cv_image)

  def GrassFollowing(self,frame,velocity,num_cont):

    kernel = np.ones((5,5),np.uint8)

    twist = Twist()
    twist.linear.x = velocity

    height, width, _ = frame.shape
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    max_height = 360 # 270
    bot_height = 60
    roi_frame = hsv_frame[height-max_height:height-bot_height, 0:width]
    centroid_x = width / 2
    centroid_y = height / 2

    # Define the lower and upper bounds for sides of the path
    lower = np.array([25, 30, 180])   
    upper = np.array([60, 90, 255]) 

    # Create a mask for path sides and remove noise
    mask = cv2.inRange(roi_frame, lower, upper)
    # mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 1)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = np.zeros((max_height-bot_height, width, 1), dtype=np.uint8)

    if len(contours) >= num_cont:
        # Find the contour with the largest area (assuming it's the path)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Remove the largest contour to find the second largest contour (assume other side of path)
        for i in range(0, num_cont):
           cv2.drawContours(max_contour, contours, i, (255, 255, 255), thickness=-1)

        max_contour = cv2.dilate(max_contour, kernel, iterations = 8)

        # m1 = cv2.moments(max_contour)
        # if m1['m00'] != 0:
        #    centroid_x = int(m1['m10'] / m1['m00'])
        #    centroid_y = int(m1['m01'] / m1['m00'])
        
        contours, _ = cv2.findContours(max_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) >= 2:
            # Find the contour with the largest area (assuming it's the path)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Remove the largest contour to find the second largest contour (assume other side of path)
            max_contour1 = contours[0]
            max_contour2 = contours[1]

            # Find the centroid of the largest contours
            M1 = cv2.moments(max_contour1)
            M2 = cv2.moments(max_contour2)

            if M1['m00'] != 0 and M2['m00'] != 0:
                cx1 = int(M1['m10'] / M1['m00'])
                cy1 = int(M1['m01'] / M1['m00'])
                cx2 = int(M2['m10'] / M2['m00'])
                cy2 = int(M2['m01'] / M2['m00'])

                centroid_x = int(np.average([cx1, cx2]))
                centroid_y = int(np.average([cy1, cy2]))

    # cv2.imshow("Contour window", max_contour)
    # cv2.waitKey(3)

    # cv2.imshow("Mask", mask)
    # cv2.waitKey(3)

    error = int(width/2) - centroid_x

    #PID well i guess only P
    P = 0.02 # 0.02 is really good
    I = 0.0125
    min_error = 25

    if np.abs(error) > min_error:
      twist.angular.z = P * error - I * (error - self.past_error)
    else: 
      twist.angular.z = 0

    self.past_error = error
    
    return twist
  
  def HillFollowing(self,frame,velocity,num_cont):

    kernel = np.ones((5,5),np.uint8)

    twist = Twist()
    twist.linear.x = velocity

    height, width, _ = frame.shape
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    max_height = 360 # 270
    bot_height = 60
    roi_frame = hsv_frame[height-max_height:height-bot_height, 0:width]
    centroid_x = width / 2
    centroid_y = height / 2

    # Define the lower and upper bounds for sides of the path
    lower = np.array([25, 30, 155])   
    upper = np.array([60, 90, 255]) 

    # Create a mask for path sides and remove noise
    mask = cv2.inRange(roi_frame, lower, upper)
    # mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 1)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = np.zeros((max_height-bot_height, width, 1), dtype=np.uint8)

    if len(contours) >= num_cont:
        # Find the contour with the largest area (assuming it's the path)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Remove the largest contour to find the second largest contour (assume other side of path)
        for i in range(0, num_cont):
           cv2.drawContours(max_contour, contours, i, (255, 255, 255), thickness=-1)

        max_contour = cv2.dilate(max_contour, kernel, iterations = 8)

        # m1 = cv2.moments(max_contour)
        # if m1['m00'] != 0:
        #    centroid_x = int(m1['m10'] / m1['m00'])
        #    centroid_y = int(m1['m01'] / m1['m00'])
        
        contours, _ = cv2.findContours(max_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) >= 2:
            # Find the contour with the largest area (assuming it's the path)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Remove the largest contour to find the second largest contour (assume other side of path)
            max_contour1 = contours[0]
            max_contour2 = contours[1]

            # Find the centroid of the largest contours
            M1 = cv2.moments(max_contour1)
            M2 = cv2.moments(max_contour2)

            if M1['m00'] != 0 and M2['m00'] != 0:
                cx1 = int(M1['m10'] / M1['m00'])
                cy1 = int(M1['m01'] / M1['m00'])
                cx2 = int(M2['m10'] / M2['m00'])
                cy2 = int(M2['m01'] / M2['m00'])

                centroid_x = int(np.average([cx1, cx2]))
                centroid_y = int(np.average([cy1, cy2]))

    # cv2.imshow("Contour window", max_contour)
    # cv2.waitKey(3)

    # cv2.imshow("Mask", mask)
    # cv2.waitKey(3)

    error = int(width/2) - centroid_x

    #PID well i guess only P
    P = 0.02 # 0.02 is really good
    I = 0.0125
    min_error = 25

    if np.abs(error) > min_error:
      twist.angular.z = P * error - I * (error - self.past_error)
    else: 
      twist.angular.z = 0

    self.past_error = error
    
    return twist

  # Follow Yoda when detected
  def yoda_follow(self,frame):

    twist = Twist()
    twist.linear.x = 0.4

    height, width, _ = frame.shape
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cx = 0

    # Define the lower and upper bounds for sides of the path
    lower = np.array([130, 0, 0])
    upper = np.array([180, 75, 75])

    # Create a mask for path sides and remove noise
    mask = cv2.inRange(hsv_frame, lower, upper)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Remove the largest contour to find the second largest contour (assume other side of path)
        max_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(max_contour) > 6000:
           return self.stop_robot()  
        # Find the centroid of the largest contour
        M = cv2.moments(max_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

    error = int(width/2) - cx

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
  
  # Wait until Yoda is detected and nearby
  def detect_yoda(self):
    largest_yoda_contour = 10
    last_contour = 1
    while not rospy.is_shutdown() and (last_contour/largest_yoda_contour > 0.9 or largest_yoda_contour < 3000):
      hsv_frame = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
      cx = 0

      # Define the lower and upper bounds for brown
      lower = np.array([130, 0, 0])
      upper = np.array([180, 75, 75])

      # Create a mask for the gray path
      mask = cv2.inRange(hsv_frame, lower, upper)

      # Find contours in the mask
      contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      if contours:
          # Find the contour with the largest area (assuming it's the path)
          max_contour = max(contours, key=cv2.contourArea)
          max_contour_area = cv2.contourArea(max_contour)
          
          if max_contour_area > largest_yoda_contour:
             largest_yoda_contour = max_contour_area

          last_contour = max_contour_area
          # print("Largest:", largest_yoda_contour, "Last:", last_contour)
      
      # cv2.imshow("YodaCam", mask)
      cv2.waitKey(3)
    
    self.yoda_found = True

  # Hard-coded Tunnel Sequence
  def tunnel(self):
    tunnel_start_time = rospy.get_time()
    clueboard_start_count = self.clueboard_count
    while rospy.get_time() - tunnel_start_time < 1:
      try:
        self.vel_pub.publish(self.GrassFollowing(self.cv_image, 0, 2))
      except CvBridgeError as e:
        print(e)

    self.vel_pub.publish(self.forward_robot())
    rospy.sleep(0.5)

    self.vel_pub.publish(self.rotate_left())
    rospy.sleep(0.2)

    tunnel_start_time = rospy.get_time()
    while rospy.get_time() - tunnel_start_time < 2:
      try:
        self.vel_pub.publish(self.GrassFollowing(self.cv_image, 0, 2))
      except CvBridgeError as e:
        print(e)

    self.vel_pub.publish(self.rotate_left())
    rospy.sleep(0.02)

    self.vel_pub.publish(self.forward_robot())
    rospy.sleep(2.5)

    cb_past_error = 0
    out_of_tunnel_time = rospy.get_time()

    while rospy.get_time() - out_of_tunnel_time < 8:
      try:
        self.vel_pub.publish(self.HillFollowing(self.cv_image, 0.25, 6))
      except CvBridgeError as e:
        print(e) 

    while rospy.get_time() - out_of_tunnel_time < 30:
      try:
        self.vel_pub.publish(self.GrassFollowing(self.cv_image, 0.25, 6))
      except CvBridgeError as e:
        print(e) 

    print("time elapsed")

    while self.clueboard_count == clueboard_start_count and not rospy.is_shutdown():
        height, width, _ = self.cv_image.shape
        hsv_frame = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)


        # Define the lower and upper bounds for blue line
        lower = np.array([118, 50, 20]) 
        upper = np.array([122, 255, 255]) 

        # Create a mask for the crosswalk line
        mask = cv2.inRange(hsv_frame, lower, upper)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
          max_contour_area = cv2.contourArea(max(contours, key=cv2.contourArea))
          if max_contour_area > 10000:
              print("MCA:", max_contour_area)
                  # Find the centroid of the largest contour
              M = cv2.moments(max(contours, key=cv2.contourArea))
              if M['m00'] != 0:
                  cx = int(M['m10'] / M['m00'])
                  cy = int(M['m01'] / M['m00'])

              #cv2.imshow("Mask window", gray_mask)
              #cv2.waitKey(3)

              error = 640 - cx

              #PID well i guess only P
              P = 0.020
              I = 0.010
              min_error = 25

              twist = Twist()
              twist.linear.x = 0.3

              if np.abs(error) > min_error:
                pass
                twist.angular.z = P * error - I * (error - cb_past_error)
              else: 
                twist.angular.z = 0

              cb_past_error = error
              
              try:
                self.vel_pub.publish(twist)
              except CvBridgeError as e:
                print(e)
          else:
              try:
                self.vel_pub.publish(self.GrassFollowing(self.cv_image, 0.3, 6))
              except CvBridgeError as e:
                print(e)
        else: 
          try:
            self.vel_pub.publish(self.GrassFollowing(self.cv_image, 0.3, 6))
          except CvBridgeError as e:
            print(e) 
      
        self.find_clueboard(self.cv_image)

    self.finished_flag = True
    
  
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

    return twist

  # Return a twist object which makes the robot turn left
  def rotate_left(self):
      twist = Twist()
      twist.angular.z = 2
      twist.linear.x = 0
      return twist
  
  def rotate_right(self):
      twist = Twist()
      twist.angular.z = -2
      twist.linear.x = 0
      return twist
  
  # Teleports the robot to the start of the course
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
      while (not rospy.is_shutdown()) and (rospy.get_time() - self.start_time < MAX_TIME) and self.finished_flag == False:

        # Show camera feed
        # cv2.imshow("Image window", self.cv_image)
        cv2.waitKey(3)

        # Line following
        if self.clueboard_count < 3:
          try:
            self.vel_pub.publish(self.RoadFollowing(self.cv_image))
          except CvBridgeError as e:
            print(e)
        elif self.pink_line_count == 0:
          try:
            self.vel_pub.publish(self.truck_detect(self.cv_image))
          except CvBridgeError as e:
            print(e)
        elif self.pink_line_count == 1: 
           # TODO move the grass following code here
          if self.past_cb5 == False and rospy.get_time() - self.pink1_time > 23 and self.clueboard_count == 4:
             spin_start_time = rospy.get_time()
             while rospy.get_time() - spin_start_time < 5.25:
                self.vel_pub.publish(self.rotate_left())
                self.find_clueboard(self.cv_image)
             self.past_cb5 = True
          elif self.past_cb6 == False and rospy.get_time() - self.pink1_time > 44.5 and self.clueboard_count == 5:
             spin_start_time = rospy.get_time()
             while rospy.get_time() - spin_start_time < 5.25:
                self.vel_pub.publish(self.rotate_right())
                self.find_clueboard(self.cv_image)
             self.past_cb6 = True
          try:
            self.vel_pub.publish(self.GrassFollowing(self.cv_image, 0.25, 5))
          except CvBridgeError as e:
            print(e)
        elif self.pink_line_count == 2:
          if self.yoda_found == False:
            self.vel_pub.publish(self.forward_robot())
            rospy.sleep(0.4)
            self.vel_pub.publish(self.stop_robot())
            self.detect_yoda()
          else: 
            try:
              self.vel_pub.publish(self.yoda_follow(self.cv_image))
            except CvBridgeError as e:
              print(e)
        else:
          self.tunnel()
        
        if self.pink_line_count != 2:
          self.find_clueboard(self.cv_image)

        # Check if the crosswalk has been detected, only runs if crosswalk has not been detected yet
        if self.crosswalk:
          self.detect_crosswalk(self.cv_image) # Initiate a sequence if the crosswalk is detected
        elif self.pink_line_count < 3:
          if rospy.get_time() - self.last_pink_time >= 8:
            self.detect_pink(self.cv_image)

      # End message
      self.comp_pub.publish("kappa, chungus, -1, DONE")
      
      # Stop the robot
      self.vel_pub.publish(self.stop_robot())

def main(args):

  rospy.init_node('image_converter', anonymous=True)
  rob = state_manager()

  rob.reset_position()
  rob.start()

  print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
