# inference_server

This package is a wrapper between ROS and the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). The Tensorflow Object Detection API is an open source framework built on top of TensorFlow that makes it easy to construct, train and deploy object detection models. This package helps us to use the object detection along with ROS in the form of an action_server. 

## Features:
   * ROS-TensorFlow Object Detection API Wrapper
   * Available as a ROS action_server

## ROS API
### actions
   * Inference.action
   
   ```
	---
	sensor_msgs/CompressedImage image
	int16 num_detections
	string[] classes
	float32[] scores
	sensor_msgs/RegionOfInterest[] bounding_boxes
	---
   ```

## Testing it with the Robot 
  * Run the object detection action_server with the following command:
 ```
rosrun inference_server inference_server_node.py
 ```
  * To run inference on the image published from the robot, run the action_client and send the goal using the following command:
```
rosrun actionlib axclient.py /inference_server inference_server/InferenceAction inference_server/InferenceAction
```
  * Receive the result with the following fields from the inference of the image, in a chronological order of the detection score.
	* image - Resultant image after inference from Object Detection API
	* num_detections - Number of detected objects in the inference image
	* classes - name of the class to which the object belongs (depends on the model used for the inference)
	* scores - detection scores or the confidence of the detection of the particular object as a particular class
	* bounding_boxes - bounding box of each of the detected object

