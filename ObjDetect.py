'''
Object Detection

Problem: We need to classify the individual objects in an image before coloring.
         This is so that we can choose training datasets that are closer related tot he objects within the input photo.

Reference: https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606

'''

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )