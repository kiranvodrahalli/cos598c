# This file contains more functions for solving problems 1.2 and 1.3. 

import os
import sys
from math import sqrt
from math import exp
from PIL import Image 
from PIL import ImageDraw


train_dir = "/Users/kiranv/college/3junior-year/spring2015/cos598c/project/train/"
test_dir = "/Users/kiranv/college/3junior-year/spring2015/cos598c/project/test/"
img_dir = "/Users/kiranv/college/3junior-year/spring2015/cos598c/project/images/"
new_imgs = "/Users/kiranv/college/3junior-year/spring2015/cos598c/project/mod_images/"


def avg(a, b):
	return (a + b + 0.)/2.

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
	# uses the x1< x2, y1< y2 assumption
	# need to fix this because box2 could be before box1.... move outside of class
	def overlap_area(self, box2):
		if box2.x1 > self.x2 or box2.y1 > self.y2:
			return 0.
		return (self.x2 - box2.x1)*(self.y2 - box2.y1)
	# returns the overlapping box between the two
	def overlap_box(self, box2):
		return None
	# returns the overlap score metric # used to merge boxes
	def overlap_score(self, box2):
		return 0
	def centroid(self):
		return (avg(self.x1, self.x2), avg(self.y1, self.y2))
	# Euclidean distance between centroids
	def centroid_dist(self, box2):
		c1 = self.centroid()
		c2 = box2.centroid()
		return sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

# perhaps there should be a discount based on Jaccard similarity or something..we just use sums a la Overfeat.
# even though in Overfeat, they might not have been dealing with this exact score. 

# SOME HELPER METHODS FOR ACTIONS ON TWO BOXES.
# box1 and box2 are boxes
# take averages of each coordinate
# new_score is calculated 
def merge_boxes(box1, box2, new_score):
	x1 = avg(box1.x1, box2.x1)
	x2 = avg(box1.x2, box2.x2)
	y1 = avg(box1.y1, box2.y1)
	y2 = avg(box1.y2, box2.y2)
	score = new_score
	return box(x1, y1, x2, y2, score)


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
def draw_boxes(box_dict, img_name, class_set, threshold):
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
			im.save(open(dump_dir + "mod_" + img_name + "_" + str(threshold) + ".jpg", "wb"), "JPEG")
	print "num classes relevant: " + str(len(classes_shown))
	print classes_shown


# generate all pictures redrawn with the appropriate boxes
# dumps these in the folder "mod_images" - check that to see the results. 
def boxes_on_pics():
	for root, dirs, files in os.walk(img_dir):
		for img in files:
			img_name = img.split(".jpg")[0]
			box_dict = gen_boxes(img_name)
			# here insert a function that transforms box_dict into merged_box_dict (i.e. after merging)
			draw_boxes(box_dict, img_name, set([]), 0.2)

# takes in a box_dict and for each class, merges object windows together
# THIS IS THE IMPLEMENTATION OF THE ALGORITHM FOR PART 2
def merged_box_dict(box_dict):
	# not implemented
	return dict()




