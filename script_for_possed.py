import cv2, os
import numpy as np
from PIL import Image
import sys
import random as rd

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
# faces = faceCascade.detectMultiScale(
#     gray_image,
#     scaleFactor=2,
#     minNeighbors=5,
#     minSize=(30, 30),
#     flags = cv2.cv.CV_HAAR_SCALE_IMAGE
# )


path_win = sys.argv[1]


def get_images_and_labels(path_to_data):
	train_images = []
	train_labels = []
	test_images = []
	test_labels = []
	for root, dirs, files in os.walk(path_to_data):
	    path = root.split(os.sep)
	    # print((len(path) - 1) * '---', os.path.basename(root))
	    for file in files:
	    	#print(file)
	        if (".BMP" in file or ".bmp" in file) and "infrared" in path and "without_glasses" in path:
				print(path[9])
				#print(int(path[9]))
				#print(path)

				label = int(path[9])
				image_path = root + "/" + file
				image_pil = Image.open(image_path).convert('L')
				image = np.array(image_pil, 'uint8')
				if(rd.random() > 0.5):
					train_images.append(image)
					train_labels.append(label)
				else:
					test_images.append(image)
					test_labels.append(label)
	return train_images, train_labels, test_images, test_labels	

train_images, train_labels, test_images, test_labels = get_images_and_labels(path_win)
cv2.destroyAllWindows()

# cv2.imshow("yo",images[0])
# cv2.waitKey(0)
print("train and test data")
print(len(train_images), len(train_labels), len(test_images), len(test_labels))

print("training LBPH")
recognizer_l = cv2.createLBPHFaceRecognizer()
recognizer_l.train(train_images, np.array(train_labels))
print("training eigenface")
recognizer_e = cv2.createEigenFaceRecognizer()
recognizer_e.train(train_images, np.array(train_labels))
print("training fisher")
recognizer_f = cv2.createFisherFaceRecognizer()
recognizer_f.train(train_images, np.array(train_labels))

print("starting predition...")
for i in range(0, len(test_labels)):
	predict_image = test_images[i]
	nbr_predicted_l, conf_l = recognizer_l.predict(predict_image)
	nbr_predicted_e, conf_e = recognizer_e.predict(predict_image)
	nbr_predicted_f, conf_f = recognizer_f.predict(predict_image)
	nbr_actual = test_labels[i]
	if nbr_actual == nbr_predicted_l:
		print( "{} is Correctly Recognized as {} with confidence {} by LBPH".format(nbr_actual, nbr_predicted, conf))
	else:
		print( "{} is Incorrectly Recognized as {} by LBPH".format(nbr_actual, nbr_predicted))
	if nbr_actual == nbr_predicted_e:
		print( "{} is Correctly Recognized as {} with confidence {} by eigenface".format(nbr_actual, nbr_predicted, conf))
	else:
		print( "{} is Incorrectly Recognized as {} by eigenface".format(nbr_actual, nbr_predicted))
	if nbr_actual == nbr_predicted_f:
		print( "{} is Correctly Recognized as {} with confidence {} by fisher".format(nbr_actual, nbr_predicted, conf))
	else:
		print( "{} is Incorrectly Recognized as {} by fisher".format(nbr_actual, nbr_predicted))
	cv2.imshow("Recognizing Face", predict_image)
	cv2.waitKey(1000)	
print("Done!")


# Append the images with the extension .sad into image_paths
# image_paths = [os.path.join(path_win, f) for f in os.listdir(path_win) if f.endswith('.sad')]
# image_paths = []
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
