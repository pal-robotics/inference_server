#! /usr/bin/env python

# ROS
import rospy
import actionlib

# OpenCV for images
import cv2

# rosmsg
from sensor_msgs.msg import Image, CompressedImage, RegionOfInterest
from std_msgs.msg import String, Header
import std_msgs.msg
import inference_server.msg
from inference_server.srv import *

# System
import os

# Tensorflow and numpy
import tensorflow as tf
import numpy as np

# object detection library
from object_detection_class import ObjectDetectionAPI
from PIL import Image

cv2.CV_LOAD_IMAGE_COLOR = 1

class InferenceServer():
	def __init__(self, sub_topic, pub_topic):
		self._sub_topic = sub_topic
		self._pub_topic = pub_topic

		self._as = actionlib.SimpleActionServer("inference_server",
                                            inference_server.msg.InferenceAction,
                                            execute_cb=self.execute_cb,
                                            auto_start=False)

		self.change_model_service = rospy.Service('~change_inference_model', ChangeInferenceModel, self.change_inference_model)

		self.image_sub = rospy.Subscriber(self._sub_topic, CompressedImage, self.receiveImage, queue_size=1, buff_size=1000000000)
		self.image_pub = rospy.Publisher(self._pub_topic, CompressedImage, queue_size=1, latch=True)
		self.image = CompressedImage()
		self.inference_input = []
		self.inference_image = CompressedImage()
		self.firstInference()
		self._as.start();
		rospy.loginfo("Inference action server running!!")
		rospy.loginfo("Change inference model server is online!")

	def firstInference(self):
		test_image = os.path.join(os.path.dirname(os.path.abspath(__file__)),"tiago_object_detection.jpg")
		input_image = cv2.imread(test_image)
		self.inference_output, num_detected, detected_classes, detected_scores, detected_boxes = object_detection.detect(input_image)
		self.inference_image = CompressedImage()
		self.inference_image.header.stamp = rospy.Time.now()
		self.inference_image.format = "jpeg"
		self.inference_image.data = np.array(cv2.imencode('.jpg', self.inference_output)[1]).tostring()
		self.image_pub.publish(self.inference_image)

	def receiveImage(self ,im_data):
		self.image = im_data
		np_arr = np.fromstring(im_data.data, np.uint8)
		self.inference_input = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)

	def change_inference_model(self, req):
		if req.model_name:
			res = object_detection.update_model(model_name=req.model_name)
			if res and req.reset_desired_classes_param:
				rospy.set_param('~desired_classes', ['all'])
				rospy.logwarn('The desired classes param has been reset to "all" for the new inference model')
			return ChangeInferenceModelResponse(res)
		else:
			return ChangeInferenceModelResponse(False)

	def execute_cb(self, goal):
		rospy.loginfo("Goal Received!")

		start_time = rospy.Time.now()

		result = inference_server.msg.InferenceResult()
		if not goal.input_image.data:
			self.inference_output, num_detected, detected_classes, detected_scores, detected_boxes = object_detection.detect(self.inference_input)
		else:
			np_arr = np.fromstring(goal.input_image.data, np.uint8)
			goal_inference_input = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
			self.inference_output, num_detected, detected_classes, detected_scores, detected_boxes = object_detection.detect(goal_inference_input)

		self.inference_image = CompressedImage()
		self.inference_image.header.stamp = rospy.Time.now()
		self.inference_image.format = "jpeg"
		self.inference_image.data = np.array(cv2.imencode('.jpg', self.inference_output)[1]).tostring()
		self.image_pub.publish(self.inference_image)

		result.image = self.inference_image
		result.num_detections = num_detected
		result.classes = detected_classes
		result.scores = detected_scores
		for i in range(len(detected_boxes)):
			box = RegionOfInterest()
			box.y_offset = detected_boxes[i][0]
			box.height = (detected_boxes[i][2] - detected_boxes[i][0])
			box.x_offset = detected_boxes[i][1]
			box.width = (detected_boxes[i][3] - detected_boxes[i][1])
			box.do_rectify = True
			result.bounding_boxes.append(box)

		total_inference_time = rospy.Time.now() - start_time
		total_inference_time = total_inference_time.to_sec()
		rospy.loginfo("The inference took {} seconds".format(total_inference_time));

		try:
			self._as.set_succeeded(result)
		except Exception as e:
			rospy.logerr(str(e))
			self._as.set_aborted(text=str(e))


if __name__ == '__main__':
	rospy.init_node('inference_action_server_node', anonymous=True)
	rospy.loginfo("Loading the inference models...........")
	inference_model = rospy.get_param('~model_name')
	tf_models_path = rospy.get_param('~tf_models_path')
	model_database_path = rospy.get_param('~model_database_path')

	object_detection = ObjectDetectionAPI(model_name = inference_model, path_to_the_model_database = model_database_path, path_to_tf_models = tf_models_path)

	sub_topic = "/xtion/rgb/image_raw/compressed"
	pub_topic = "/inference_image/image_raw/compressed"
	server = InferenceServer(sub_topic, pub_topic)
	rospy.spin()
