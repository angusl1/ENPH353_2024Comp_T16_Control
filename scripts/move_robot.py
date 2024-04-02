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
import numpy as np

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()

    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
    self.timer_sub = rospy.Subscriber("/clock", Clock)
    self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
    self.comp_pub = rospy.Publisher("/score_tracker", String, queue_size = 1)
    
    rospy.sleep(1)
    self.comp_pub.publish("kappa,chungus,0,ZANIEL")
    self.start_time = rospy.get_time()

  def lineFollowing(self,frame):

    twist = Twist()
    twist.linear.x = 0.3

    width, height, _ = frame.shape
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cx = 0

    # Define the lower and upper bounds for dark blue color
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask for the dark blue path
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the contour with the largest area (assuming it's the path)
        max_contour = max(contours, key=cv2.contourArea)

        # Find the centroid of the largest contour
        M = cv2.moments(max_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Draw a red circle at the centroid position
            cv2.circle(frame, (cx, cy), radius=20, color=(0, 0, 255), thickness=-1)

    #PID
    k = 3
    if ((width / 2) - cx > 0):
        twist.angular.z = k
    else: 
        twist.angular.z = -k
    
    return twist

  def callback(self,data):
      try:
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      except CvBridgeError as e:
        print(e)

  def start(self):
      while (not rospy.is_shutdown()) and (rospy.get_time() - self.start_time < 5):

        cv2.imshow("Image window", self.cv_image)
        cv2.waitKey(3)

        try:
          self.vel_pub.publish(self.lineFollowing(self.cv_image))
        except CvBridgeError as e:
          print(e)

      self.comp_pub.publish("kappa, chungus, -1, DONE")
      twist = Twist()
      twist.angular.z = 0
      twist.linear.x = 0
      self.vel_pub.publish(twist)



def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()
  ic.start()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
