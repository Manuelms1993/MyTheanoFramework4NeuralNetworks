# coding=utf-8
import fnmatch
import os
import pickle
import numpy
import PIL.Image as Image
import scipy.misc as misc

FRAME_EXTENSION = "jpeg"


def _get_all_file_paths(base_path):
	matches = []
	for root, dirnames, filenames in os.walk(base_path):
		for filename in fnmatch.filter(filenames, "*.%s" % FRAME_EXTENSION):
			matches.append(os.path.join(root, filename))
	return matches


def _get_labels(frame_path_list):
	return [label for label in [directory[len(directory)-1] for directory in [os.path.dirname(f) for f in frame_path_list]]]


def get_pickle_dictionary(base_path, percentage_test=0.8):
	"""
	percentage_test : percentage of samples for training. Default: 80%
	
	Returns two dictionaries with the CIFAR-10 format, for training and testing:
	{
		data  : [samples] x (resolution²) matrix,
		labels: [samples] long list with label numbers
	}

	Base path must be with this structure:
	├── frames_obj1
	├── frames_obj2
	├── frames_obj3
	├── frames_obj4
	├── frames_obj5
	└── ...
	
	:param base_path: Path to the frames directory
	:return: dictionary ready to be "pickled"
	"""
	frame_path_list = _get_all_file_paths(base_path)
	number_of_items = len(frame_path_list)
	numpy.random.shuffle(frame_path_list)
	
	items_for_training = int(percentage_test * number_of_items)
	
	data_matrix_train = numpy.array([channel
							   for channel in [misc.imread(image_path, flatten=True).flatten()
									for image_path in frame_path_list[0:items_for_training]]])
	label_list_train  = numpy.array([label
							   for label in _get_labels(frame_path_list[0:items_for_training])])
		 
	data_matrix_test = numpy.array([channel
							   for channel in [misc.imread(image_path, flatten=True).flatten()
									for image_path in frame_path_list[items_for_training:number_of_items]]])
	label_list_test	 = numpy.array([label
							   for label in _get_labels(frame_path_list[items_for_training:number_of_items])])							 
							   
	return ({
		"data": data_matrix_train.astype("int"),
		"labels" : label_list_train.astype("int")
	},
	{
		"data": data_matrix_test.astype("int"),
		"labels" : label_list_test.astype("int")
	})
	
	
if __name__ == "__main__":
	import sys
	
	if len (sys.argv) < 2:
		print "You should specify the path to the images.\n"
	else:
		path_to_the_images =  sys.argv[1]
		if os.access(path_to_the_images, os.F_OK) != True:
			print "The specified path is not found!!"
		else:
			if len(sys.argv) > 2:
				percentage = float(sys.argv[2])
			else:
				percentage = 0.8
				
			if percentage > 1.0 or percentage < 0.0: 
				print "Percentage should be between 0 and 1 [0% and 100%]"
			else:
				print "Serialization in progress..."
				train, test = get_pickle_dictionary(path_to_the_images , percentage)
				pickle.dump(train, open("data_objects", "wb"))
				pickle.dump(test, open("test_objects", "wb"))
				print "Serialization successfully done!!"
				print "You can find serialized files as 'data_objects' and 'test_objects'."



