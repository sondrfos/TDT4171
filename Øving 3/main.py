import random as rand
from math import log

ATTR = 7
VALUES = [1,2]
ATTRIBUTES = [i for i in range(ATTR)]
TRAIN = "training.txt"
TEST = "test.txt"


class Tree(object):

	def __init__(self, classification, label):
		self.classification = classification
		self.label = label
		self.children = []

	def addChildren(self, child):
		self.children.append(child)

	def getChildren(self):
		return self.children

	def setClassification(self, classification):
		self.classification = classification

	def getLabel(self):
		return self.label

	def getClassification(self):
		return self.classification

	def printTree(self,level):
		children = self.children
		if self.label == -1:
			print("Classification: " + str(self.classification) + " at level " + str(level))
		else:
			print("Label: " + str(self.label) + " at level " + str(level))
			level +=1
			for child in children:
				child.printTree(level)

def readExamples(file):
	instream = open(file, 'r')
	data = []
	for line in instream:
		line = line.replace("\t", "")
		line = line.replace("\n", "")
		line = map(int, line)
		data.append(line)
	instream.close()
	return data

def puralityValue(examples):
	class1, class2 = 0,0
	for example in examples:
		if example[ATTR] == 1:
			class1+=1
		if example[ATTR] == 2:
			class2 += 1
	if class1 > class2:
		return 1
	return 2

def allElementsEqualClass(examples):
	example = examples[0]
	firstClass = example[ATTR]
	for example in examples:
		if example[ATTR] != firstClass:
			return 0
	return firstClass

def randomImportance(attributes):

	return attributes[rand.randint(0,len(attributes)-1)]

def B(q):
	if q == 1:
		return 0
	if q == 0:
		return 0
	return -(q*log(q,2) + (1-q)*log((1-q),2))

def entropy(examples):
	n = float(len(examples))
	if n == 0:
		return 0
	p = 0
	for example in examples:
		if example[ATTR] == 1:
			p+=1
	return B(p/n)

def remainder(examples, attr):
	class1, class2 = [], []
	n = float(len(examples))
	for example in examples:
		if example[attr] == 1:
			class1.append(example)
		else:
			class2.append(example)
	remainder_class1 = (len(class1)/n) * entropy(class1)
	remainder_class2 = (len(class2)/n) * entropy(class2)
	return remainder_class1 + remainder_class2

def informationGainImportance(examples, attributes):
	index, value = 0,-1
	currentEntropy = entropy(examples)
	for attribute in attributes:
		informationGain = currentEntropy - remainder(examples, attribute)
		if informationGain > value:
			index = attribute
			value = informationGain
	return index
	
#Decision-tree-learning algorithm. Importance decieds the importance-function used, 0 = a random value assigned to each attribute. 1 = a information gain approach.  
def decisionTreeLearning(examples, attributes, parent_examples, importance):
	oneClassification = allElementsEqualClass(examples)
	if not examples:
		return Tree(puralityValue(parent_examples), -1)
	elif oneClassification:
		return Tree(oneClassification,-1)
	elif not attributes:
		return Tree(puralityValue(examples), -1)
	else:
		if importance ==1:
			A = informationGainImportance(examples, attributes)
		else:
			A = randomImportance(attributes)
		tree = Tree(0, A)
		attributes.remove(A)
		for value in VALUES:
			exs = []
			for example in examples:
				if example[A] == value:
					exs.append(example)
			subTree = decisionTreeLearning(exs, attributes, examples, importance)
			subTree.setClassification(value)
			tree.addChildren(subTree)
	return tree

def testTree(tree, testData):
	root = tree
	numRight = 0
	for test in testData:
		tree = root 
		label = tree.getLabel()
		while(label != -1):	
			children = tree.getChildren()
			tree = children[(test[label]-1)]
			label = tree.getLabel()
		classification = tree.getClassification()
		if classification == test[ATTR]:
			numRight += 1
	return numRight
			


importance = 1
trainingData = readExamples(TRAIN)
testData = readExamples(TEST)
tree = decisionTreeLearning(trainingData, ATTRIBUTES, [], importance)
tree.printTree(0)
print(len(testData))
print(testTree(tree, testData))