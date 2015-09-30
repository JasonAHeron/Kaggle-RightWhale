import os
import math
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import scipy
import scipy.misc
import scipy.cluster

from PIL import Image
from pylab import array
from scipy import ndimage
from scipy.misc import imsave
from scipy.ndimage import filters
from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image



source = 'data/imgs_subset/'
target = 'data/imgs/'

def get_imlist(path):
	""" Returns a list of filenames for all jpg images in a directory. """
	return [os.path.join( path, f) for f in os.listdir(path) if f.endswith('.jpg')]

def img_hist(im):
	pl.figure()
	pl.gray()
	pl.contour(im, origin='image')
	pl.axis('equal')
	pl.axis('off')
	pl.figure()
	pl.hist(im.flatten(), 128)
	pl.show()


def histeq(im, nbr_bins = 256):
	""" Histogram equalization of a grayscale image. """
	# get image histogram
	imhist, bins = pl.histogram(im.flatten(), nbr_bins, normed = True)
	cdf = imhist.cumsum()  # cumulative distribution function
	cdf = 255 * cdf / cdf[-1]  # normalize
	# use linear interpolation of cdf to find new pixel values
	im2 = pl.interp(im.flatten(), bins[:-1], cdf)
	return im2.reshape(im.shape)


def find_object(im):
	im2 = ndimage.filters.gaussian_filter(im, sigma=20)
	binary_img = im2 > 0.5

	# im2 = zeros(im.shape)
	# for i in xrange(3):
	# 	im2[:,:,i] = filters.gaussian_filter(im[:,:,i], 30)
	# im2 = uint8(im2)

	mask = (im2 > im2.mean()).astype(np.float)

	label_im, nb_labels = ndimage.label(mask)

	sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
	mask_size = sizes < 1000
	remove_pixel = mask_size[label_im]
	label_im[remove_pixel] = 0
	labels = np.unique(label_im)
	label_clean = np.searchsorted(labels, label_im)

	# fig = plt.figure(figsize=(20,5))
	# fig.add_subplot(1, 3, 1)
	# plt.imshow(im)
	# fig.add_subplot(1, 3, 2)
	# plt.imshow(im2, cmap=plt.cm.gray)
	# fig.add_subplot(1, 3, 3)
	# plt.imshow(label_clean)
	# plt.show()

def display_im(name, imlist, rows=1, columns=1, gray=True):
	if gray:
		plt.gray()
	fig = plt.figure(figsize=(20,5))
	if rows == 1:
		columns = len(imlist)
	for r in xrange(rows):
		for c in xrange(columns):
			fig.add_subplot(rows, columns, r*columns + c+1)
			plt.imshow(imlist[r*columns+c])
	plt.draw()
	plt.savefig('data/processed/%s.png' % name, bbox_inches='tight');
	plt.close()


def most_frequent_colour(image):
	w, h = image.size
	pixels = image.getcolors(w * h)
	most_frequent_pixel = pixels[0]
	for count, colour in pixels:
		if count > most_frequent_pixel[0]:
			most_frequent_pixel = (count, colour)
	# fig = plt.figure(figsize=(1,1))
	# ax1 = fig.add_subplot(111)
	# print most_frequent_pixel[1]
	p = most_frequent_pixel[1]
	# color = (p[0]/255., p[1]/255., p[2]/255.)
	# ax1.add_patch(patches.Rectangle((0.1, 0.1), 0.5, 0.5, color=color, fill=True))
	# plt.show()

	return p


def denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):
    """ An implementation of the Rudin-Osher-Fatemi (ROF) denoising model
        using the numerical procedure presented in Eq. (11) of A. Chambolle
        (2005). Implemented using periodic boundary conditions.

        Input: noisy input image (grayscale), initial guess for U, weight of
        the TV-regularizing term, steplength, tolerance for the stop criterion

        Output: denoised and detextured image, texture residual. """

    m,n = im.shape #size of noisy image

    # initialize
    U = U_init
    Px = np.zeros((m, n)) #x-component to the dual field
    Py = np.zeros((m, n)) #y-component of the dual field
    error = 1

    while (error > tolerance):
        Uold = U

        # gradient of primal variable
        GradUx = np.roll(U,-1,axis=1)-U # x-component of U's gradient
        GradUy = np.roll(U,-1,axis=0)-U # y-component of U's gradient

        # update the dual varible
        PxNew = Px + (tau/tv_weight)*GradUx # non-normalized update of x-component (dual)
        PyNew = Py + (tau/tv_weight)*GradUy # non-normalized update of y-component (dual)
        NormNew = np.maximum(1,np.sqrt(PxNew**2+PyNew**2))

        Px = PxNew/NormNew # update of x-component (dual)
        Py = PyNew/NormNew # update of y-component (dual)

        # update the primal variable
        RxPx = np.roll(Px,1,axis=1) # right x-translation of x-component
        RyPy = np.roll(Py,1,axis=0) # right y-translation of y-component

        DivP = (Px-RxPx)+(Py-RyPy) # divergence of the dual field.
        U = im + tv_weight*DivP # update of the primal variable

        # update of error
        error = np.linalg.norm(U-Uold)/np.sqrt(n*m);

    return U,im-U # denoised image and texture residual


def blur_color(im):
	im2 = np.zeros(im.shape)
	for i in xrange(3):
		im2[:,:,i] = filters.gaussian_filter(im[:,:,i], 1)

	return np.uint8(im2)


def crop(source, target):
	imlist = get_imlist(source)

	kernel_size = 3
	scale = 1
	delta = 0
	ddepth = cv2.CV_16S

	for ii in xrange(25,30):
		impath = imlist[ii]
		filename = impath.split('/')[-1]
		print "Process file %s" % filename

		im = cv2.imread(impath)

		im2 = cv2.GaussianBlur(im,(3,3),0)
		im2 = cv2.resize(im2, (300, 300))
		hsv = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
		# loop over the boundaries
		# create NumPy arrays from the boundaries
		lower = np.array([340, 1.18, 80])
		upper = np.array([340, 1.18, 100])

		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(hsv, lower, upper)
		output = cv2.bitwise_and(im2, im2, mask=mask)

		# show the images
		cv2.imshow("images", np.hstack([im2, output]))
		cv2.waitKey(0)
		# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		# gray = cv2.equalizeHist(gray)

		# gray_lap = cv2.Laplacian(gray, ddepth, ksize=kernel_size, scale=scale, delta=delta)
		# dst = cv2.convertScaleAbs(gray_lap)

		# cv2.imshow('laplacian', im2)
		# cv2.waitKey(0)
		cv2.destroyAllWindows()








crop(source, target)
