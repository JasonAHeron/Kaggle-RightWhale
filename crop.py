import os
source = 'data/imgs_subset/'
target = 'data/imgs/'

def get_imlist(path):
	""" Returns a list of filenames for all jpg images in a directory. """
	return [os.path.join( path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def crop(source, target):
	imlist = get_imlist(source)


crop(source, target)
