import numpy as np
import matplotlib.pyplot as plt
import cv2

IMAGE = "soldiers_1941.jpg"

prototxt = "./model/colorization_deploy_v2.prototxt"
model = "./model/colorization_release_v2.caffemodel"
points = "./model/pts_in_hull.npy"
image =  "./input_images/"+IMAGE

net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)