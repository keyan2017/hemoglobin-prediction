# hemoglobin-prediction
Non-invasive hemoglobin concentration prediction based on patient eyelid images

The prediction of hemoglobin concentration is divided into two steps: first, the eye image is cropped to extract the eyelid part, and then the concentration is predicted using the eyelid area. The cropping part of the code borrows from the mask-rcnn segmentation model in MMDetection of openlab.The concentration prediction part is based on MobileNetV3.In addition, comparisons were made with other models

Modify the configuration file path file（in config.py） to run.



Thanks for your attention!