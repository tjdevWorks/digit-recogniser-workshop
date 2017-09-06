import numpy as np
import base64
from PIL import Image

def preprocess(jpgtxt):
		# data = base64.decodestring(data)
		data = jpgtxt.split(',')[-1]
		data = base64.b64decode(data.encode('ascii'))

		g = open("temp.jpg", "wb")
		g.write(data)
		g.close()

		pic = Image.open("temp.jpg")

		M = np.array(pic) #now we have image data in numpy

		M = rgb2gray(M)
		M = squareTrim(M,threshold=0)
		M = naiveInterp2D(M,28,28)
		[N, mean, sigma] = normalize(M)
		n = N.reshape(-1)
		if np.isnan(np.sum(n)):
			n = np.zeros(n.shape)
		return n

def squareTrim(M, min_side=28, threshold=0):
	assert M.shape[0]==M.shape[1],"Input matrix must be a square"
	wsum = np.sum(M,axis=0)
	nonzero = np.where(wsum > threshold*M.shape[1])[0]
	if len(nonzero) >=1:
		wstart = nonzero[0]
		wend = nonzero[-1]
	else:
		wstart=0 ; wend = 0

	hsum = np.sum(M,axis=1)
	nonzero = np.where(hsum > threshold*M.shape[0])[0]
	if len(nonzero) >=1:
		hstart = nonzero[0]
		hend = nonzero[-1]
	else:
		hstart=0 ; hend = 0

	diff = abs((wend-wstart) - (hend-hstart))
	if (wend-wstart > hend-hstart):
		side = max(wend-wstart+1, min_side)
		m = np.zeros((side, side))
		cropped = M[hstart:hend+1,wstart:wend+1]
		shift = int(diff/2)
		m[shift:cropped.shape[0]+shift,:cropped.shape[1]] = cropped
	else:
		side = max(hend-hstart+1, min_side)
		m = np.zeros((side, side))
		cropped = M[hstart:hend+1,wstart:wend+1]
		shift=int(diff/2)
		m[:cropped.shape[0],shift:cropped.shape[1]+shift] = cropped
	return m

def rgb2gray(rgb):
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return gray

def naiveInterp2D(M, newx, newy):
	result = np.zeros((newx,newy))
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			indx = int(i*newx / M.shape[0])
			indy = int(j*newy / M.shape[1])
			result[indx,indy] +=M[i,j]
	return result

def normalize(M):
	sigma = np.std(M)
	mean = np.mean(M)
	return [(M-mean)/sigma, mean, sigma]
