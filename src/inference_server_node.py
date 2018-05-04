#! /usr/bin/env python

# ROS
import rospy
import actionlib

# OpenCV for images
from cv_bridge import CvBridge, CvBridgeError
import cv2

# rosmsg
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Header
import std_msgs.msg
import inference_server.msg

# System
import sys
import os

# Tensorflow and numpy
import tensorflow as tf
import numpy as np

# object detection library
import obj_det_fun 

bridge = CvBridge()

cv2.CV_LOAD_IMAGE_COLOR = 1

class InferenceServer():
	def __init__(self, sub_topic, pub_topic):
		self.detec_graph, self.category = obj_det_fun.objdetect()
		self.sess = tf.Session(graph = self.detec_graph)
		rospy.loginfo('Tensorflow inference session successfully created!!')

		self._sub_topic = sub_topic
		self._pub_topic = pub_topic

		self._as = actionlib.SimpleActionServer("inference_server",
                                            inference_server.msg.InferenceAction,
                                            execute_cb=self.execute_cb,
                                            auto_start = False)

		self.image_sub = rospy.Subscriber(self._sub_topic, CompressedImage, self.receiveImage, queue_size = 1, buff_size=1000000000)
		self.image_pub = rospy.Publisher(self._pub_topic, CompressedImage, queue_size=1)
		rospy.loginfo("Inference action server running!!")
		self.image = CompressedImage()
		self.inference_input = []
		self.inference_image = CompressedImage()
		self._as.start();

	def receiveImage(self ,im_data):
		self.image = im_data
		np_arr = np.fromstring(im_data.data, np.uint8)
		self.inference_input = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
		self.image_pub.publish(self.inference_image)

	def execute_cb(self, goal):
		rospy.loginfo("Goal Received!")
		result = inference_server.msg.InferenceResult()
		self.inference_output = obj_det_fun.detect(self.sess, self.detec_graph, self.category, self.inference_input)

		self.inference_image = CompressedImage()
		self.inference_image.header.stamp = rospy.Time.now()
		self.inference_image.format = "jpeg"
		self.inference_image.data = np.array(cv2.imencode('.jpg', self.inference_output)[1]).tostring()
		self.image_pub.publish(self.inference_image)

		result.image = self.inference_image
		try:
			self._as.set_succeeded(result)
		except Exception as e:
			rospy.logerr(str(e))
			self._as.set_aborted(text=str(e))


if __name__ == '__main__':
	rospy.init_node('inference_action_server_node', anonymous=True)
        rospy.loginfo("Loading the inference models...........")
	obj_det_fun.loadmodel()

	sub_topic = "/xtion/rgb/image_rect_color/compressed"
#	sub_topic = "/usb_cam/image_raw/compressed"
	pub_topic = "/inference_image/image_raw/compressed"
	server = InferenceServer(sub_topic, pub_topic)
	rospy.spin()
