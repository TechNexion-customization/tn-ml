import numpy as np
import argparse
import time
import cv2
import os
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,help="path to Caffe pre-trained model")

args = vars(ap.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

time.sleep(2.0)

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

frame = cv2.imread("cat.jpg")
frame_rz = imutils.resize(frame,width=32)
blob = cv2.dnn.blobFromImage(frame_rz, 0.003921, (32,32), (0.4914, 0.4822, 0.4465))
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
preds = preds.reshape((1,len(classes)))
idx = np.argsort(preds[0])[::-1][0]
text = "Label: {}".format(classes[idx])
cv2.putText(frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.imwrite("output.jpg", frame);
os.system("sudo -E chromium --no-sandbox output.jpg")
