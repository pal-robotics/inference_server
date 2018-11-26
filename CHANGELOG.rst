^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package inference_server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.0.2 (2018-11-26)
------------------
* Merge branch 'optional_image' into 'master'
  added an optional image as an input to Inference action
  See merge request saikishor/inference_server!8
* added an optional image as an input to Inference action
* Merge branch 'fixing_cv_bgr' into 'master'
  Fixing CV Image BGR Issue
  See merge request saikishor/inference_server!7
* removing unused cvbridge and sys headers
  (cherry picked from commit 110a3ee9e6616692199fff438190ff4b609e13aa)
* fixing BGR issue with cv2
  (cherry picked from commit 15e362f962b04bb2ad28cac0b125e968c638d395)
* Merge branch 'change_model_service' into 'master'
  add service to update with new inference model
  See merge request saikishor/inference_server!5
* add service to update with new inference model
* Contributors: Sai Kishor Kothakota, Victor Lopez, saikishor

0.0.1 (2018-08-09)
------------------
* Merge branch 'inference_time' into 'master'
  Inference_time and subscriber changes
  See merge request saikishor/inference_server!4
* reverting subscriber changes
* added inference_time logging
* Merge branch 'desired_classes_param' into 'master'
  Setting desired classes using param server
  See merge request saikishor/inference_server!3
* README.md Update
* added support to choose the desired classes from param server
* Merge branch 'locations_as_params' into 'master'
  Locations as params
  See merge request saikishor/inference_server!2
* fixing the unavailable model download location
* passing inference details through param server
* Merge branch 'add-tensorflow-link' into 'master'
  Update README.md
  See merge request saikishor/inference_server!1
* Update README.md
* Removed the unnecessary subscription of images and added timeout for receiving images
* update README.md
* add README.md
* correcting bounding_boxes typo in action
* removed unused headers
* object_detection class update and inference_server_ndoe update
* added RegionofInterest action msg and initial inference image
* InferenceAdvanced Action and Object Detection Class update
* add gpu allow_growth config
* inference_server initial commit
* reset repository
* image action_server
* Contributors: Victor Lopez, saikishor
