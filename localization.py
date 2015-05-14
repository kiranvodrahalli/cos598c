import os
from math import sqrt
from math import exp

train_dir = "/Users/kiranv/college/3junior-year/spring2015/cos598c/project/train/"
test_dir = "/Users/kiranv/college/3junior-year/spring2015/cos598c/project/test/"
img_dir = "/Users/kiranv/college/3junior-year/spring2015/cos598c/project/images/"

def avg(a, b):
	return (a + b + 0.)/2.
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
	def overlap_area(self, box2):
		if box2.x1 > self.x2 or box2.y1 > self.y2:
			return 0.
		return (self.x2 - box2.x1)*(self.y2 - box2.y1)
	def centroid(self):
		return (avg(self.x1, self.x2), avg(self.y1, self.y2))
	# Euclidean distance between centroids
	def centroid_dist(self, box2):
		c1 = self.centroid()
		c2 = box2.centroid()
		return sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

# box1 and box2 are boxes
# take averages of each coordinate
# add scores
def merge_boxes(box1, box2):
	x1 = avg(box1.x1, box2.x1)
	x2 = avg(box1.x2, box2.x2)
	y1 = avg(box1.y1, box2.y1)
	y2 = avg(box1.y2, box2.y2)
	score = box1.score + box2.score # for the lolz - this is not really used. we need to calculate
									# a different score for the boxes based on the merging step.
	return box(x1, y1, x2, y2, score)

# returns a dict from class name to list of boxes. 
# note that given scores are log(prob). need to exp them to get products. 
# perhaps there should be a discount based on Jaccard similarity or something..we just use sums a la Overfeat.
# even though in Overfeat, they might not have been dealing with this exact score. 
def gen_boxes(img_name):
	trd = train_dir + img_name + "/"
	tst = test_dir + img_name + "/"
	print trd
	print tst
	box_dict = dict()
	for root, dirs, files in os.walk(trd):
		for f in files: 
			class_name = f.split(".txt")[0]
			if class_name not in box_dict:
				box_dict[class_name] = []
			for line in open(os.path.join(root, f), "rb"):
				line = map(lambda r: float(r), line.split(" "))
				print line
				print "======================================"
				assert(len(line) == 5)
				# score given is log(probility)
				box_dict[class_name].append(box(line[0], line[1], line[2], line[3], exp(line[4])))
	for root, dirs, files in os.walk(tst):
		for f in files:
			class_name = f.split(".txt")[0]
			if class_name not in box_dict:
				box_dict[class_name] = []
			for line in open(os.path.join(root, f), "rb"):
				line = map(lambda r: float(r), line.split(" "))
				assert(len(line) == 5)
				# score given is log(prob)
				box_dict[class_name].append(box(line[0], line[1], line[2], line[3], exp(line[4])))
	return box_dict


