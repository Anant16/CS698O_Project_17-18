import cv2, os
import numpy as np
from PIL import Image
import sys
cascadePath = "../haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
# faces = faceCascade.detectMultiScale(
#     gray_image,
#     scaleFactor=2,
#     minNeighbors=5,
#     minSize=(30, 30),
#     flags = cv2.cv.CV_HAAR_SCALE_IMAGE
# )

recognizer = cv2.createLBPHFaceRecognizer()

path_to_data = "../../datasets/yalefaces/yalefaces"
path_to_thermal = "../datasets/thermal"
#recognizer = cv2.createLBPHFRecognizer()
path_win = sys.argv[1]


def get_images_and_labels(path):
	image_paths = [os.path.join(path, f) \
	for f in os.listdir(path) if not f.endswith('.sad')]
	#print(image_paths)
	images = []
	labels = []

	for image_path in image_paths:
		image_pil = Image.open(image_path).convert('L')
		image = np.array(image_pil, 'uint8')
		# get the labels
		#print(image)
		label = int(os.path.split(image_path)[1].split('.')[0].replace('subject',""))
		# detect faces
		# print("detecting in " + image_path)
		faces = faceCascade.detectMultiScale( image)
		for (x, y, w, h) in faces:
			# images.append(image[y: y+h, x: x+w])
			images.append(image)
			labels.append(label)

			print("crop_" + image_path.split('/')[-1])
			cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
			cv2.imshow("Adding this face to training set..", image)
			cv2.imwrite("crop_" + image_path.split('/')[-1] + ".jpg", image[y: y+h, x: x+w])
			cv2.waitKey(50) # set 50
	return images, labels

images, labels = get_images_and_labels(path_to_data)
cv2.destroyAllWindows()

# cv2.imshow("yo",images[0])
# cv2.waitKey(0)


# recognizer.train(images, np.array(labels))

# Append the images with the extension .sad into image_paths
# image_paths = [os.path.join(path_to_data, f) for f in os.listdir(path_to_data) if f.endswith('.sad')]
# for image_path in image_paths:
# 	predict_image_pil = Image.open(image_path).convert('L')
# 	predict_image = np.array(predict_image_pil, 'uint8')
# 	faces = faceCascade.detectMultiScale(predict_image)
# 	for (x, y, w, h) in faces:
# 		# nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
# 		nbr_predicted, conf = recognizer.predict(predict_image)		
# 		nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
# 		if nbr_actual == nbr_predicted:
# 			print( "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf))
# 		else:
# 			print( "{} is Incorrectly Recognized as {}".format(nbr_actual, nbr_predicted))
# 		cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
# 		cv2.waitKey(1000)