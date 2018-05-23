import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import rospy
import getpass 

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

#username = getpass.getuser()
TF_MODELS_PATH = '/TFM/models/research/object_detection'
MODEL_DATABASE = '/TFM/model_database'
		
# This is needed to access files from research folder of Tensorflow models API
sys.path.append(TF_MODELS_PATH)	

from utils import label_map_util

from utils import visualization_utils as vis_util
	
# What model to download or use.
#MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
MODEL_NAME = 'ssd_inception_v2_items'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

#local path directory of all models
MODEL_PATH = os.path.join(MODEL_DATABASE, MODEL_FILE)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_DATABASE + '/' + MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join(TF_MODELS_PATH + '/data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join(MODEL_DATABASE, MODEL_NAME, 'items_label_map.pbtxt')

#NUM_CLASSES = 90
NUM_CLASSES = 3

def loadmodel():
	if os.path.exists(PATH_TO_CKPT):
	    rospy.logwarn('Found and verified .pb file : ' +  PATH_TO_CKPT)	
	    return
	if os.path.exists(MODEL_PATH):
	    rospy.logwarn('Found and verified : ' +  MODEL_PATH)
	    tar_file = tarfile.open(MODEL_PATH)
	else:
	    rospy.logwarn('The model is not available at the model database location: '+ MODEL_PATH)
	    rospy.logwarn('Attempting to download over internet:' + MODEL_FILE) 
	    opener = urllib.request.URLopener()
	    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	    rospy.logwarn('\nDownload Complete!')
	    tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
	  file_name = os.path.basename(file.name)
	  if 'frozen_inference_graph.pb' in file_name:
	    #choose this option only if you want to extract in current work directory cwd
#		    tar_file.extract(file, os.getcwd())
	    tar_file.extract(file, MODEL_DATABASE)

def objdetect():
	detection_graph = tf.Graph()
	with detection_graph.as_default():
	  od_graph_def = tf.GraphDef()
	  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
	    serialized_graph = fid.read()
	    od_graph_def.ParseFromString(serialized_graph)
	    tf.import_graph_def(od_graph_def, name='')
	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)
	return detection_graph, category_index

def load_image_into_numpy_array(image):
#	(im_width, im_height) = image.size
	(im_width, im_height, _) = image.shape
#	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
	return np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)

def detect(sess, detection_graph,category_index, data):
	IMAGE_SIZE = (12, 8)
	with detection_graph.as_default():
		# Definite input and output Tensors for detection_graph
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		# Each box represents a part of the image where a particular object was detected.
		detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')
		# the array based representation of the image data into the variable, this will be used later in order to prepare the
		# result image with boxes and labels on it.
		image_np = data
#		image_np = load_image_into_numpy_array(image)
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)
		# Actual detection.
		(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: image_np_expanded})

		# Minimum score threshold
		minimum_threshold = 0.55 #default value is 0.5 (50%) used by the API

		# Visualization of the results of a detection.
		vis_util.visualize_boxes_and_labels_on_image_array(
		image_np,
		np.squeeze(boxes),
		np.squeeze(classes).astype(np.int32),
		np.squeeze(scores),
		category_index,
		use_normalized_coordinates=True,
		min_score_thresh=minimum_threshold,
		line_thickness=8)

		# Extraction of the detection scores, classes, boxes and number
		boxes = np.squeeze(boxes)
		scores = np.squeeze(scores)
		classes = np.squeeze(classes).astype(np.int32)

		detected_boxes = []
		detected_classes = []
		detected_scores = []
		num_detected = 0
		(im_width, im_height,_) = image_np.shape
		for i in range(int(boxes.shape[0])):
			if scores[i] > minimum_threshold:
				num_detected = num_detected + 1
				box = (boxes[i].tolist())
				box[0] = int(np.ceil(box[0]*im_width))
				box[2] = int(np.ceil(box[2]*im_width))
				box[1] = int(np.ceil(box[1]*im_height))
				box[3] = int(np.ceil(box[3]*im_height))
				detected_boxes.append(tuple(box))
				detected_classes.append(category_index.get(classes[i]).get('name').encode('utf8'))
				detected_scores.append(scores[i])

#		print('detected classes : {}, detected_scores : {}, num_detections : {}, detected boxes : {}'.format(detected_classes, detected_scores, num_detected, detected_boxes))
#		print('box type {} , shape {}'.format(type(detected_boxes), len(detected_boxes)))
		return image_np, num_detected, detected_classes, detected_scores, detected_boxes
