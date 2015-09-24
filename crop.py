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

	for ii in xrange(25,30):
		impath = imlist[ii]
		filename = impath.split('/')[-1]
		print "Process file %s" % filename

		im = array(Image.open(impath))
		c = most_frequent_colour(Image.open(impath))
		c = array(c)
		# im2 = np.power(im-c, 2)
		# im2 = np.zeros(im.shape)
		# for i in xrange(im.shape[0]):
		# 	for j in xrange(im.shape[1]):
		# 		for k in xrange(im.shape[2]):
		# 			im2[i,j,k] = im[i,j,k]-c[k]
		im2 = im
		binary_image = np.where(im > np.mean(im), 1.0, 0.0)
		im3 = filters.gaussian_filter(binary_image, 3)
		display_im(filename, [im, im, binary_image, im3], 2, 2)
		continue

		# im_gray = array(Image.open(impath).convert('L'))
		im = filters.gaussian_filter(im, 3)
		im = histeq(im)

		im_gray = array(Image.fromarray(np.uint8(im)).convert('L'))
		im_gray = filters.gaussian_filter(im_gray, 3)

		im2 = 255 - im_gray # invert image
		im3 = (100.0/ 255) * im_gray # clamp to interval 100... 200
		im4 = 255.0 * (im_gray/ 255.0)** 2 # squared

		# Gaussian derivative filters
		# sigma = 2
		# imx = np.zeros(im_gray.shape)
		# filters.gaussian_filter(im_gray, (sigma, sigma), (0, 1), imx)

		# imy = np.zeros(im_gray.shape)
		# filters.gaussian_filter(im_gray, (sigma, sigma), (1, 0), imy)

		# magnitude = np.sqrt(imx**2+imy**2)


		# img_hist(im)

		# im2, cdf = histeq(im)
		# img_hist(im2)
		# find_object(im_gray)


		# display_im(filename, [im_gray, im2, im3, im4], 2, 2)
		# display_im(filename + 'der', [magnitude])
		# display_im(filename + 'blur', [im, blur_color(im)])
		# U, T = denoise(im_gray, im_gray)
		# display_im(filename + 'noise', [im_gray, U])







crop(source, target)
