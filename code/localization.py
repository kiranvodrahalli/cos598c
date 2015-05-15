# This file contains more functions for solving problems 2.2 and 2.3. 

import os
import sys
from math import exp
from PIL import Image 
from PIL import ImageDraw
from scipy.cluster.vq import whiten, kmeans
import numpy as np
from numpy.linalg import norm


train_dir = "/Users/kiranv/college/3junior-year/spring2015/cos598c/project/data/train/"
test_dir = "/Users/kiranv/college/3junior-year/spring2015/cos598c/project/data/test/"
img_dir = "/Users/kiranv/college/3junior-year/spring2015/cos598c/project/data/images/"
new_imgs = "/Users/kiranv/college/3junior-year/spring2015/cos598c/project/results/mod_images/"


def avg(a, b):
	return (a + b + 0.)/2.
#--------------------------------------------------------------------#
# THE BOX CLASS
# assume x1 < x2, y1 < y2 - they are diagonals
class box:
	def __init__(self, x1, y1, x2, y2, score):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		self.score = score
	def __str__(self):
		return "(" + str(self.x1) + ", " + str(self.y1) + ", " + str(self.x2) + ", " + str(self.y2) + "): " + str(self.score)
	def area(self):
		return abs(self.x2 - self.x1)*abs(self.y2 - self.y1)
	# pt = (x, y)
	def inside(self, pt):
		if (pt[0] <= self.x2 and pt[0] >= self.x1) or (pt[0] >= self.x2 and pt[0] <= self.x1):
			if (pt[1] <= self.y2 and pt[1] >= self.y1) or (pt[1] >= self.y2 and pt[1] <= self.y1):
				return True
		return False
	def centroid(self):
		return np.array([avg(self.x1, self.x2), avg(self.y1, self.y2)])

# perhaps there should be a discount based on Jaccard similarity or something..we just use sums a la Overfeat.
# even though in Overfeat, they might not have been dealing with this exact score. 
#--------------------------------------------------------------------#
# SOME HELPER METHODS FOR ACTIONS ON TWO BOXES.

# uses the x1< x2, y1< y2 assumption
# need to fix this because box2 could be before box1.... move outside of class
def overlap_area(box1, box2):
	if box1.x1 <= box2.x1 and box2.x1 <= box1.x2:
		if box1.y1 <= box2.y1 and box2.y1 <= box1.y2:
			return (box1.x2 - box2.x1)*(box1.y2 - box2.y1)
		elif box2.y1 <= box1.y1 and box1.y1 <= box2.y2:
			return (box1.x2 - box2.x1)*(box2.y2 - box1.y1)
		else:
			return 0.
	elif box2.x1 <= box1.x1 and box1.x1 <= box2.x2:
		if box1.y1 <= box2.y1 and box2.y1 <= box1.y2:
			return (box2.x2 - box1.x1)*(box1.y2 - box2.y1)
		elif box2.y1 <= box1.y1 and box1.y1 <= box2.y2:
			return (box2.x2 - box1.x1)*(box2.y2 - box1.y1)
		else:
			return 0.
	return 0.

# Euclidean distance between centroids
def centroid_dist(box1, box2):
	c1 = box1.centroid()
	c2 = box2.centroid()
	return norm(c1 - c2)
# box1 and box2 are boxes
# take averages of each coordinate
# the new score is the sum is calculated 
def merge_boxes(box1, box2):
	x1 = avg(box1.x1, box2.x1)
	x2 = avg(box1.x2, box2.x2)
	y1 = avg(box1.y1, box2.y1)
	y2 = avg(box1.y2, box2.y2)
	score = box1.score + box2.score # sum, as described in the paper
	return box(x1, y1, x2, y2, score)

#--------------------------------------------------------------------#
# PARSING THE OBJECT FILES TO DRAW BOXES ON IMAGES

# returns a dict from class name to list of boxes. 
# we apply exp to the score to get everything positive. 
def gen_boxes(img_name):
	trd = train_dir + img_name + "/"
	tst = test_dir + img_name + "/"
	print trd
	print tst
	box_dict = dict()
	for root, dirs, files in os.walk(trd):
		for f in files: 
			if f.split(".")[1] == "txt":
				class_name = f.split(".txt")[0]
				if class_name not in box_dict:
					box_dict[class_name] = []
				for line in open(os.path.join(root, f), "rb"):
					print os.path.join(root, f)
					print line
					print "=------="
					line = map(lambda r: float(r), line.split(" "))
					print line
					print "======================================"
					assert(len(line) == 5)
					# score given is log(probility)
					box_dict[class_name].append(box(line[0], line[1], line[2], line[3], exp(line[4])))
	for root, dirs, files in os.walk(tst):
		for f in files:
			print f
			if f.split(".")[1] == "txt":
				class_name = f.split(".txt")[0]
				if class_name not in box_dict:
					box_dict[class_name] = []
				for line in open(os.path.join(root, f), "rb"):
					line = map(lambda r: float(r), line.split(" "))
					assert(len(line) == 5)
					box_dict[class_name].append(box(line[0], line[1], line[2], line[3], exp(line[4])))
	return box_dict

# A list of office type objects that we particularly consider for the given dataset. 
office_class_list = ["chair", "books", "wall", "clock", "door", "drawer", "shelves", "screen", "window", "floor", "showcase"]
office_class_set = set(office_class_list)

# Names of the colors we assign to the objects
color_names = ["crimson", "deeppink2", "purple", "dodgerblue", "lightsteelblue", "green4", "darkorange", "springgreen4", "yellow", "ivory4", "darkgreen"]
colors = ["#DC143C", "#EE1289", "#800080", "#1E90FF", "#CAE1FF", "#008B00", "#FF8C00", "#00EE76", "#FFFF00", "#8B8B83", "#006400"]
# dictionary map from objects to colors for drawing
color_map = dict(zip(office_class_list, colors))
# dictionary map from objects to color names for printing out. 
colorname_map = dict(zip(office_class_list, color_names))

print "Color Map: "
print colorname_map
# draws the bounding boxes on an image given a dictionary of classes to lists of boxes. 
# class_set specifies which classes are to be drawn.
# threshold marks the minimum score a bounding box must have to be drawn.
# it should be > 0. 
# combiner_name is either empty string (base case), "greedy", or "kmeans"
def draw_boxes(box_dict, img_name, class_set, threshold, combiner_name):
	img_path = img_dir + img_name + ".jpg"
	show = True
	classes_shown = set()
	for obj_class in box_dict:
		color = ""
		# empty set denotes everything, for convenience.
		if len(class_set) > 0:
			if obj_class not in class_set:
				show = False
			else:
				show = True
		if show:
			im = Image.open(img_path)
			draw = ImageDraw.Draw(im)
			if obj_class not in color_map:
				color = "#ffffff"
			else:
				color = color_map[obj_class]
			boxes = box_dict[obj_class]
			print obj_class + ": " + str(len(boxes))
			print "Threshold: " + str(threshold)
			for box in boxes:
				if box.score > threshold:
					classes_shown.add(obj_class)
					print obj_class + ": " + str(box)
					# thickness of the lines is dependent on the confidence of the box
					draw.line((box.x1, box.y1, box.x1, box.y2), fill= color, width=int(box.score*5))
					draw.line((box.x1, box.y1, box.x2, box.y1), fill= color, width=int(box.score*5))
					draw.line((box.x2, box.y2, box.x2, box.y1), fill= color, width=int(box.score*5))
					draw.line((box.x2, box.y2, box.x1, box.y2), fill= color, width=int(box.score*5))
			print "================================"
			dump_dir = new_imgs + obj_class + "/"
			if not os.path.exists(dump_dir):
				print "Making directory " + dump_dir
				os.makedirs(dump_dir)
			im.save(open(dump_dir + "mod_" + img_name + "_" + combiner_name + "_" + str(threshold) + ".jpg", "wb"), "JPEG")
	print "num classes relevant: " + str(len(classes_shown))
	print classes_shown


# generate all pictures redrawn with the appropriate boxes
# dumps these in the folder "mod_images" - check that to see the results. 
# combiner is a function that somehow reduces the set of box proposals into only a few boxes.
# the "combiner methods" are defined below (this is essentially the last part of Q2 and Q3)
def boxes_on_pics(combiner):
	for root, dirs, files in os.walk(img_dir):
		for img in files:
			img_name = img.split(".jpg")[0]
			box_dict = gen_boxes(img_name)
			# here insert the function that transforms box_dict into merged_box_dict (i.e. after merging)
			box_dict = combiner(box_dict)
			draw_boxes(box_dict, img_name, set([]), 0.2)


#================================================================================================
# COMBINERS
# FOR THE FOLLOWING TWO FUNCTIONS, TO GET THE IMAGES, SIMPLY CALL
# >> boxes_on_pics(merged_box_dict)
# OR
# >> boxes_on_pics(apply_kmeans)
#--------------------------------------------------------------------#
# takes in a box_dict and for each class, merges object windows together
# THIS IS THE IMPLEMENTATION OF THE ALGORITHM IN QUESTION 2
# threshold limits the number of pairs to search (if there are 100000 proposals, this argmin is intractable)
# suggests their algorithm is either underspecified or inefficient for general case.
# t is the stopping threshold for the score: stop when > t. 
def merged_box_dict(box_dict, thresh, t):
	# for every object class in the box_dict
	#    reduce the list of boxes to the merged boxes using paper's greedy algorithm
	# return the new merged dictionary
	assert(t > 0)
	merged_dict = dict()
	for obj_class in box_dict:
		boxes = box_dict[obj_class]
		pair_set = set()
		count = 0
		# build set of box pairs that is reasonably sized
		for i in range(len(boxes)):
			b1 = boxes[i]
			for j in range(i+1, len(boxes)):
				b2 = boxes[j]
				if count < thresh:
					pair_set.add((b1, b2))
					count += 1
		# implement the algorithm from the paper
		# what we are trying to minimize
		def match_score(b1, b2):
			# as directly specified in the paper
			return centroid_dist(b1, b2) + overlap_area(b1, b2)
		score_dict = dict()
		# we want the minimum (argmin)
		matchval = float('inf')
		minpair = None
		print len(pair_set)
		# init loop to calculate all scores for pairs one time.
		for pair in pair_set:
				if pair not in score_dict:
					new_score = match_score(pair[0], pair[1])
					score_dict[pair] = new_score
					print new_score
					if new_score < matchval:
						matchval = new_score
						minpair = pair
		print "===================START OF MAIN LOOP======================="
		# the main loop 
		while matchval < t and len(pair_set) > 0:
			print len(pair_set)
			# merge boxes
			new_box = merge_boxes(minpair[0], minpair[1])
			# remove the minimum pair
			pair_set.remove(minpair)
			# remove all pairs that contained either of the boxes in the minimum pair
			rem_list = []
			# this is actually 2 * the size of the box list, not box list^2 in terms of append operations
			for pair in pair_set:
				if pair[0] == minpair[0] or pair[0] == minpair[1]:
					rem_list.append(pair)
				if pair[1] == minpair[0] or pair[1] == minpair[1]:
					rem_list.append(pair)
			for pair in rem_list:
				pair_set.remove(pair)
			# add the merged box and add its scores to the dict
			# the point is to avoid double looping every time if possible.
			# we avoid looping over old pairs this way: 2*|box list| instead of |box list|^2.
			# reset minpair to avoid key errors.. we need the minimum THIS time around.
			# also reset matchval 
			matchval = float('inf')
			minpair = None
			to_add = []
			for pair in pair_set:
				# add the two new pairs for each other box currently in the set
				newpair1 = (new_box, pair[0])
				newpair2 = (new_box, pair[1])
				to_add.append(newpair1)
				to_add.append(newpair2)
				# update score_dict with the new_box
				if newpair1 not in score_dict:
					new_score = match_score(new_box, pair[0])
					score_dict[newpair1] = new_score
					print new_score
					if new_score < matchval:
						matchval = new_score
						minpair = newpair1
				if newpair2 not in score_dict:
					new_score = match_score(new_box, pair[1])
					score_dict[newpair2] = new_score
					print new_score
					if new_score < matchval:
						matchval = new_score
						minpair = newpair2
			for pair in to_add:
				pair_set.add(pair)
		new_box_set = set()
		for pair in pair_set:
			new_box_set.add(pair[0])
			new_box_set.add(pair[1])
		# change it into a list
		boxes_list = []
		for box in new_box_set:
			boxes_list.append(box)
		# updated the merged_dict, now that we have our merged boxes
		if obj_class not in merged_dict:
			merged_dict[obj_class] = boxes_list
	return merged_dict


#--------------------------------------------------------------------#
# this is the implementation of K-means for QUESTION 3
# takes in a box_dict and converts it. k is the k in k-means. (k clusters)
def apply_kmeans(box_dict, k):
	# for every object class in the box_dict
	#     reduce the list of boxes to the clustered boxes with kmeans
	# return the new dictionary
	kmeans_dict = dict()
	for obj_class in box_dict:
		print obj_class
		boxes = box_dict[obj_class]
		if len(boxes) > k:
			# write a representation for each proposal box as a vector
			def box_to_vec(pbox):
				# list of metrics which we want to reduce the Euclidean distance of:
				# includes centroid, and each of the individual coordinates of the box,
				# which are used to recover box coordinates after the k means in vector reprepresentation
				# are found. To weight the impact of the centroid measure, 
				# we multiply by 1/area: the centroid matters less as box area increases. 
				# we also include the coordinates, since distances between them are relevant as well. 
				# Note that including the original coordinates in the vector allows us to recover the 
				# original representation of the box. 
				# we also include the score (scaled down) for the same reason. We scale it down since score-space
				# should not really affect the distance between boxes (having similar scores is not necessarily a good reason
				# to combine or not)
				metrics = [pbox.centroid()[0], pbox.centroid()[1], pbox.centroid()[0]/pbox.area(), pbox.centroid()[1]/pbox.area(), pbox.x1, pbox.y1, pbox.x2, pbox.y2, 0.00001*pbox.score]
				return metrics
			# we will append the columns together and then take transpose
			# so that each row is a box with n features (here n = 9)
			first_col = box_to_vec(boxes[0])
			# for rescaling
			oldx1, oldy1, oldx2, oldy2, oldscore = first_col[4], first_col[5], first_col[6], first_col[7], first_col[8]
			first_col = np.array(first_col)
			first_col = first_col.T
			box_mat = first_col
			for i in range(1, len(boxes)):
				new_col = np.array(box_to_vec(boxes[i]))
				new_col = new_col.T
				box_mat = np.c_[box_mat, new_col]
			box_mat = box_mat.T
			box_mat = box_mat.astype('float')
			# whiten 
			box_mat = whiten(box_mat)
			# need to rescale the coords when we recover the boxes from the representation vectors
			newx1, newy1, newx2, newy2, newscore = 0, 0, 0, 0, 0
			if len(np.shape(box_mat)) > 1:
				newx1, newy1, newx2, newy2, newscore = box_mat[0][4], box_mat[0][5], box_mat[0][6], box_mat[0][7], box_mat[0][8]
			else:
				newx1, newy1, newx2, newy2, newscore = box_mat[4], box_mat[5], box_mat[6], box_mat[7], box_mat[8]
			scalex1, scaley1, scalex2, scaley2, scalescore = oldx1/(0. + newx1), oldy1/(0. + newy1), oldx2/(0. + newx2), oldy2/(0. + newy2), oldscore/(0. + newscore)
			# use k-means
			codebook, distortion = kmeans(box_mat, k)
			centroid_boxes = []
			for i in range(np.shape(codebook)[0]):
				# we chop off from 4 onwards because these are (pbox.x1, pbox.y1, pbox.x2, pbox.y2, pbox.score)
				# this is a direct inverse from box_to_vec
				# need to multiply these coords by standard deviations across all instances of feature.
				thebox = box(scalex1*codebook[i][4], scaley1*codebook[i][5], scalex2*codebook[i][6], scaley2*codebook[i][7], scalescore* codebook[i][8])
				centroid_boxes.append(thebox)
			print "# of centroids: " + str(len(centroid_boxes))
			print centroid_boxes[0]
			print centroid_boxes[1]
			print centroid_boxes[2]
			if obj_class not in kmeans_dict:
				kmeans_dict[obj_class] = []
			kmeans_dict[obj_class] = centroid_boxes
		else:
			kmeans_dict[obj_class] = box_dict[obj_class]
		print "==================================="
	return kmeans_dict


