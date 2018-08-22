import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import rospy
import yaml

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

class ObjectDetectionAPI():
	def __init__(self, model_name = "ssd_inception_v2_coco_11_06_2017", path_to_the_model_database = "/TFM/model_database", path_to_tf_models = "/TFM/models"):
		self.model_name = model_name
		self.path_to_the_model_database = path_to_the_model_database
		self.path_to_tf_models = path_to_tf_models

		# This function looks for the models in the location MODEL_DATABASE, if not it downloads them over internet
		# if they are standard models
		self.look_for_the_model(self.model_name, self.path_to_the_model_database)

		# Loads the detection graph and the category index from the extracted info from the look_for_the_model method
		self.load_detection_graph_and_category_index()

		self.required_classes = []

		# This config helps to run the tensorflow without having some memory issues
		# For instance, when cuda is not able to allocate some memory.
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth = True
		self.session = tf.Session(config = self.config, graph = self.detection_graph)
		rospy.loginfo('Tensorflow inference session successfully created!!');

	def __del__(self):
		self.session.close()
		rospy.loginfo('Tensorflow inference session successfully closed!!')

	def look_for_the_model(self, model_name, path_to_the_model_database):
		file_suffix = ".tar.gz"
		self.model_path = os.path.join(path_to_the_model_database, model_name)
		self.model_file = self.model_path + file_suffix
		#self.path_to_the_inference_graph = os.path.join(path_to_the_model_database, model_name, 'frozen_inference_graph.pb')
		#self.path_to_label_map_file = self.model_path + "/*_label_map.pbtxt"
		self.path_to_yaml = os.path.join(self.model_path, 'config.yaml')
		if os.path.exists(self.path_to_yaml):
			with open(self.path_to_yaml, "r") as file:
				config_data = yaml.load(file)
				self.path_to_label_map_file = os.path.join(path_to_the_model_database, model_name, config_data["label_map_file"])
				self.path_to_the_inference_graph = os.path.join(path_to_the_model_database, model_name, config_data["inference_graph"])

			if os.path.exists(self.path_to_the_inference_graph) and os.path.exists(self.path_to_label_map_file):
				rospy.logwarn('Found and verified .pb file : ' +  self.path_to_the_inference_graph)	
				rospy.logwarn('Found the label_map file : ' +  os.path.basename(self.path_to_label_map_file))
				return
		else:
			self.path_to_label_map_file = os.path.join(self.path_to_tf_models, 'research/object_detection/data', 'mscoco_label_map.pbtxt')
			self.path_to_the_inference_graph = os.path.join(path_to_the_model_database, model_name, 'frozen_inference_graph.pb')
			if os.path.exists(self.path_to_the_inference_graph) and os.path.exists(self.path_to_label_map_file):
				rospy.logwarn('Found and verified .pb file : ' +  self.path_to_the_inference_graph)	
				rospy.logwarn('Found the label_map file : ' +  os.path.basename(self.path_to_label_map_file))
				return

		if os.path.exists(self.model_file):
			rospy.logwarn('Found and verified : ' +  self.model_file)
			tar_file = tarfile.open(self.model_file)
		else:
			rospy.logwarn('The model is not available at the model database location: '+ self.model_file)
			rospy.logwarn('Attempting to download over internet:' + model_name + file_suffix) 
			opener = urllib.request.URLopener()
			opener.retrieve(DOWNLOAD_BASE + model_name + file_suffix, self.model_file)
			rospy.logwarn('Download Complete!\n')
			tar_file = tarfile.open(self.model_file)

		# Getting the config yaml
		for file in tar_file.getmembers():
			file_name = os.path.basename(file.name)
			if 'config.yaml' in file_name:
				##choose this option only if you want to extract in current work directory cwd
				#tar_file.extract(file, os.getcwd())
				tar_file.extract(file, path_to_the_model_database)
				self.path_to_yaml = os.path.join(self.model_path, 'config.yaml')
				with open(self.path_to_yaml, "r") as file:
					config_data = yaml.load(file)
					self.path_to_label_map_file = os.path.join(path_to_the_model_database, model_name, config_data["label_map_file"])
					self.path_to_the_inference_graph = os.path.join(path_to_the_model_database, model_name, config_data["inference_graph"])
					break

		for file in tar_file.getmembers():
			file_name = os.path.basename(file.name)
			if os.path.basename(self.path_to_the_inference_graph) in file_name:
				tar_file.extract(file, path_to_the_model_database)
			if os.path.basename(self.path_to_label_map_file) in file_name:
				tar_file.extract(file, path_to_the_model_database)

	def load_detection_graph_and_category_index(self):
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(self.path_to_the_inference_graph, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
		label_map = label_map_util.load_labelmap(self.path_to_label_map_file)
		self.num_classes = label_map_util.get_max_label_map_index(label_map)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_classes, use_display_name=True)
		self.category_index = label_map_util.create_category_index(categories)
		self.label_map_dict = label_map_util.get_label_map_dict(self.path_to_label_map_file, use_display_name=True)
		#return detection_graph, category_index

	def get_detection_graph_and_category_index(self, model_name, path_to_the_model_database):
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(self.path_to_the_inference_graph, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
		label_map = label_map_util.load_labelmap(self.path_to_label_map_file)
		self.num_classes = label_map_util.get_max_label_map_index(label_map)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_classes, use_display_name=True)
		category_index = label_map_util.create_category_index(categories)
		label_map_dict = label_map_util.get_label_map_dict(self.path_to_label_map_file, use_display_name=True)
		return detection_graph, category_index, label_map_dict

	def update_model(self, model_name = "ssd_inception_v2_coco_11_06_2017", path_to_the_model_database = None):
		self.model_name = model_name
		if not (path_to_the_model_database is None):
			self.path_to_the_model_database = path_to_the_model_database
		self.look_for_the_model(self.model_name, self.path_to_the_model_database)
		detection_graph, category_index, label_map_dict = self.get_detection_graph_and_category_index(self.model_name, self.path_to_the_model_database)
		self.create_new_session(detection_graph, category_index, label_map_dict)
		return True

	def load_image_into_numpy_array(self, image):
		(im_width, im_height) = image.size
		return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
		#(im_width, im_height, _) = image.shape
		#return np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)

	def create_new_session(self, new_detection_graph, new_category_index, new_label_map_dict):
		# the session need to be closed before, in order to release memory for other session
		self.session.close()
		self.detection_graph = new_detection_graph
		self.category_index = new_category_index
		self.label_map_dict = new_label_map_dict
		self.session = tf.Session(config=self.config, graph = self.detection_graph)
		rospy.loginfo('Tensorflow new inference session successfully created!!');

	def replace_session(self, session, new_detection_graph, new_category_index):
		rospy.logwarn('You are replacing the sessions of the API, please be sure to close the session to release memory');
		self.detection_graph = new_detection_graph
		self.category_index = new_category_index
		self.session = session
		rospy.loginfo('Tensorflow inference session successfully replaced!!');

	def detect(self, data):
		with self.detection_graph.as_default():
			# Definite input and output Tensors for self.detection_graph
			image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
			# the array based representation of the image data into the variable, this will be used later in order to prepare the
			# result image with boxes and labels on it.
			image_np = data
			#image_np = load_image_into_numpy_array(image)

			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			# image_np[...,::-1] converts the BGR image to RGB image, as cv2's default color scheme is BGR
			# The inference result from the BGR image wouldn't be as expected compared to RGB, RGB scheme is always preferred for inference
			image_np_expanded = np.expand_dims(image_np[...,::-1], axis=0)
			# Actual detection.
			(boxes, scores, classes, num) = self.session.run(
			[detection_boxes, detection_scores, detection_classes, num_detections],
			feed_dict={image_tensor: image_np_expanded})

			# Minimum score threshold
			minimum_threshold = 0.55 #default value is 0.5 (50%) used by the API

			# Extraction of the detection scores, classes, boxes and number
			boxes = np.squeeze(boxes)
			scores = np.squeeze(scores)
			classes = np.squeeze(classes).astype(np.int32)

			# This helps to set the list of desired class detection alone or all classes detection from the param server
			class_selection = rospy.get_param('~desired_classes', ['all'])
			if not (set(self.required_classes) == set(class_selection)):
				self.required_classes = class_selection
				self.desired_classes = class_selection
				rospy.loginfo("{} class detection is set using param server".format(self.desired_classes))
				for req_class in self.desired_classes:
					if self.label_map_dict.get(req_class) is None and not req_class == 'all':
						rospy.logwarn("The class {} is not found in the provided model lable map and is being removed from your desired class list".format(req_class))
						rospy.logerr("The following classes are only present in the provided label map : \n {}".format(self.label_map_dict.keys()))
						self.desired_classes.remove(req_class)

				if not self.desired_classes:
					rospy.logerr("May be the specified classes are not present in the label map, resetting it to all classes!!")
					self.desired_classes = ['all']

			indices = []
			if not 'all' in self.desired_classes:
				for class_index in self.desired_classes:
					try:
						class_indices = np.intersect1d(np.where(classes == int(self.label_map_dict[class_index]))[0], np.where(scores > minimum_threshold)[0])
					except KeyError, e:
						rospy.logwarn("KeyError for class {} - The desired class might not exist in the new label map".format(class_index))
						rospy.logwarn("Resetting the desired_classes param to ['all']")
						rospy.set_param('~desired_classes', ['all'])
						indices = range(0, len(self.label_map_dict))
						break
					except:
						raise
					if not class_indices is None and not len(class_indices) == 0:
						indices.extend(class_indices)

				if not indices is None and not len(indices) == 0:
					# Visualization of the results of a detection.
					vis_util.visualize_boxes_and_labels_on_image_array(
					image_np,
					boxes[indices,:],
					classes[indices],
					scores[indices],
					self.category_index,
					use_normalized_coordinates=True,
					min_score_thresh=minimum_threshold,
					line_thickness=8)

			else:
				# Visualization of the results of a detection.
				vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				boxes,
				classes,
				scores,
				self.category_index,
				use_normalized_coordinates=True,
				min_score_thresh=minimum_threshold,
				line_thickness=8)
				#indices = range(int(boxes.shape[0]))
				indices = np.where(scores > minimum_threshold)[0]

			detected_boxes = []
			detected_classes = []
			detected_scores = []
			num_detected = 0
			(self.im_width, self.im_height,_) = image_np.shape
			for i in indices:
				num_detected = num_detected + 1
				box = (boxes[i].tolist())
				box[0] = int(np.ceil(box[0]*self.im_width))
				box[2] = int(np.ceil(box[2]*self.im_width))
				box[1] = int(np.ceil(box[1]*self.im_height))
				box[3] = int(np.ceil(box[3]*self.im_height))
				detected_boxes.append(tuple(box))
				detected_classes.append(self.category_index.get(classes[i]).get('name').encode('utf8'))
				detected_scores.append(scores[i])

			return image_np, num_detected, detected_classes, detected_scores, detected_boxes
