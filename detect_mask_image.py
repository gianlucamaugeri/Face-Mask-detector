# USAGE
# python detect_mask_image.py --image examples/example_01.png

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image, ImageDraw

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model(args["model"])

# load the input image from disk, clone it, and grab the image spatial
# dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# detect face
frames_tracked = []
im_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
boxes, _ = mtcnn.detect(im_pil)
frame_draw = im_pil.copy()
draw = ImageDraw.Draw(frame_draw)
for box in boxes:
	startX, startY, endX, endY = box.astype("int")
	(startX, startY) = (max(0, startX), max(0, startY))
	(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
	face = image[startY:endY, startX:endX]
	face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
	face = cv2.resize(face, (224, 224))
	face = img_to_array(face)
	face = preprocess_input(face)
	face = np.expand_dims(face, axis=0)

	# pass the face through the model to determine if the face
	# has a mask or not
	(mask, withoutMask) = model.predict(face)[0]

	# determine the class label and color we'll use to draw
	# the bounding box and text
	label = "Mask" if mask > withoutMask else "No Mask"
	color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

	# include the probability in the label
	label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

	# display the label and bounding box rectangle on the output
	# frame
	cv2.putText(image, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
	cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)