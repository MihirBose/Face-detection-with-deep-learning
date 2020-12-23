import numpy as np
import argparse
import cv2
import os
import sys
import json
from os import listdir
from os.path import isfile, join

# References:-
# https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
# http://caffe.berkeleyvision.org/model_zoo.html

#folder_path = "test folder/images"

# Read only the images from provided folder path
folder_path = sys.argv[1]
Files = [cv2.imread(join(folder_path, files)) for files in listdir(folder_path) if isfile(join(folder_path, files)) and cv2.imread(join(folder_path, files)) is not None]
image_names = [files for files in listdir(folder_path) if isfile(join(folder_path, files)) and cv2.imread(join(folder_path, files)) is not None]

json_list = [] # Add the detected faces box parameters here

# Path for the model files
prototxt_path = "Model_Files/prototxt.txt"
model_path = "Model_Files/pre-trained.caffemodel"

# Create the network object
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Loop over the images
for index, image in enumerate(Files):
    count = 0

    # Preprocessing
    (height, width) = image.shape[:2]
    resized_img = cv2.resize(image, (224,224))
    image_blob = cv2.dnn.blobFromImage(resized_img, 1.0, (224, 224), (104.0, 117.0, 123.0)) 

    # Set the input
    net.setInput(image_blob)
    # Perform forward pass on the cnn
    objs_detected = net.forward()

    t = objs_detected.shape[2]
    for i in range(0, t):
        conf = objs_detected[0, 0, i, 2]
        # Find coordinates of bounding box if detected as a face with >= 90% probability
        if conf >= 0.9:
            rect = objs_detected[0, 0, i, 3:7] * np.array([width, height, width, height])
            x_topleft = int(rect[0])
            y_topleft = int(rect[1])
            x_bottomright = int(rect[2])
            y_bottomright = int(rect[3])

            element = {"iname":image_names[index], "bbox":[x_topleft, y_topleft, x_bottomright, y_bottomright]}
            json_list.append(element)

# Write the output to a json file
output_json = join(folder_path, "results.json")
with open(output_json, 'w') as f:
    json.dump(json_list, f)