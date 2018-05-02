#! /usr/bin/env python

import rospy

import actionlib
import std_msgs.msg
from sensor_msgs.msg import CompressedImage
import imagenet_server.msg
import sys

class ImageNetServer():
	def __init__(self, sub_topic):
		self._sub_topic = sub_topic

		self._as = actionlib.SimpleActionServer("image_net_server",
                                            imagenet_server.msg.ImageNetAction,
                                            execute_cb=self.execute_cb,
                                            auto_start = False)

		self.image_sub = rospy.Subscriber(self._sub_topic, CompressedImage, self.receiveImage, queue_size = 1, buff_size=1000000000)
		rospy.loginfo("ImageNet action server running.")
		self.image = CompressedImage()
		self._as.start();

	def receiveImage(self ,im_data):
		self.image = im_data

	def execute_cb(self, goal):
		rospy.loginfo("Goal Received!")
		result = imagenet_server.msg.ImageNetResult()
		result.image = self.image
		try:
			self._as.set_succeeded(result)
		except Exception as e:
			rospy.logerr(str(e))
			self._as.set_aborted(text=str(e))

if __name__ == '__main__':
	rospy.init_node('ImageNet_action_server_node')
	sub_topic = "/xtion/rgb/image_rect_color/compressed"
#	sub_topic = "/usb_cam/image_raw/compressed"
	server = ImageNetServer(sub_topic)
	rospy.spin()
