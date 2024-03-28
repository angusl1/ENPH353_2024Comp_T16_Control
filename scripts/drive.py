#!/usr/bin/env python3

# Copied code for node from http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist


class drive_forward:

    def __init__(self):
        # we want to subscribe to the image that is published automatically by the camera
        # then we want to publish the velocity which is automatically heard by the robot
        # self.image_pub = rospy.Publisher("image_topic_2", Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image)
        self.timer_sub = rospy.Subscriber("/clock", Clock)
        self.publish = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
        self.comp_pub = rospy.Publisher("/score_tracker", String, queue_size = 1)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        print('loop!')

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
    
        velocity.linear.x = 1
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
