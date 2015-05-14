# This file contains methods to reverse the file setup, as described
# in problem 1.1

import os
from shutil import copyfile

train_dir = "/home/knv/train"
train_dump = "/home/knv/cos598c/train"

test_dir = "/home/knv/test"
test_dump = "/home/knv/cos598c/test"

#which = tr or tst
# for a specific image, this gets all the object location files and flips them.
# which stands for train or test, since an image may be assigned to either.
def get_files(img, which):
	d = ""
	o = ""
	if which == "tr":
		d = train_dir
		o = train_dump
	else:
		d = test_dir
		o = test_dump
	dump_dir = o + "/" + img
	if not os.path.exists(dump_dir):
		print "Making directory " + dump_dir
		os.makedirs(dump_dir)
	for subdir, dirs, files in os.walk(d):
		for f in files:
			if f.strip(".txt") == img:
				real_file = os.path.join(subdir, f)
				fname = os.path.join(subdir, f.strip(".txt"))
				fname = fname.split(d)[1].strip("/")
				fname = fname.split("/")[0]
				fname = dump_dir + "/" + fname + ".txt"
				print "Dumping " + real_file + " in " + fname
				copyfile(real_file, fname) 
				print "-------------------------------"

imgs = "/home/knv/cos598c/images"
# applies the previous function for every single image. 
def make_all():
	for subdir, dirs, files in os.walk(imgs):
		for img in files:
			img_name = img.split(".jpg")[0]
			print "Getting training files for " + img_name + "..."
			get_files(img_name, "tr")
			print "Getting testing files for " + img_name + "..."
			get_files(img_name, "tst")
			print "======================================================="

# need to delete the empty directories
def remove_empty_dirs():
	d1 = train_dump
	d2 = test_dump
	for subdir, dirs, files in os.walk(d1):
		if not files:
			if subdir != d1 and subdir != train_dir and subdir != test_dir:
				print subdir
				os.rmdir(subdir)
	for subdir, dirs, files in os.walk(d2):
		if not files:
			if subdir != d2 and subdir != train_dir and subdir != test_dir:
				print subdir
				os.rmdir(subdir)
