import cv2, os
import numpy as np
from PIL import Image
from matplotlib.mlab import PCA

# Partially implemented preprocessing
def preprocessing(image):
	#image is np.array
	image_size = (80, 100)
	#image = cv2.resize(image, image_size)
	# median blur/filter
	median_blur_amount = 1
	image = cv2.medianBlur(image, median_blur_amount)
	# zero mean normalize
	norm_im = image.astype(np.float32)
	norm_im = (norm_im - norm_im.mean()) / norm_im.std() #zero mean unit variance
	# DOG
	g_blur_tuple = (5,5)
	g_blur_amount1 = 1.0
	g_blur_amount2 = 0.5
	gblur1 = cv2.GaussianBlur(norm_im, g_blur_tuple, g_blur_amount1)
	gblur2 = cv2.GaussianBlur(norm_im, g_blur_tuple, g_blur_amount2)
	image_dog = gblur2 - gblur1
	# HOG Descriptors:
	winSize = image_size
	blockSize = (20,20)
	blockStride = (20,20)
	cellSize = (10,10)
	nbins = 9
	derivAperture = 1
	winSigma = 4.
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 0
	nlevels = 64
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
	                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
	#compute(img[, winStride[, padding[, locations]]]) -> descriptors
	winStride = (8,8)
	padding = (8,8)
	locations = (image_size,)
	hist = hog.compute(image,winStride,padding, locations)
	return hist

# wrong abhi
def princomp(A, numpc=0):
	# computing eigenvalues and eigenvectors of covariance matrix
	M = (A-np.mean(A.T,axis=1)).T # subtract the mean (along columns)
	print(np.cov(M).shape)
	[latent,coeff] = np.linalg.eig(np.cov(M))
	p = np.size(coeff,axis=1)
	idx = np.argsort(latent) # sorting the eigenvalues
	idx = idx[::-1]       # in ascending order
	# sorting eigenvectors according to the sorted eigenvalues
	coeff = coeff[:,idx]
	latent = latent[idx] # sorting eigenvalues
	if numpc < p and numpc >= 0:
		coeff = coeff[:,range(numpc)] # cutting some PCs if needed
	score = np.dot(coeff.T,M) # projection of the data in the new space
	return coeff,score,latent

if __name__ == '__main__':
	import sys
	filename = sys.argv[1]
	image_pil = Image.open(filename)#.convert('L')
	image = np.array(image_pil)
	hist = preprocessing(image)
	print(hist.shape)
	A = np.zeros((2,hist.shape[0]))
	A[0]=hist.T
	A[1]=3*hist.T+2
	print(A.shape)
	# print(A)
	#coeff,score,latent = princomp(A,64)
	# results = PCA(A)
	
