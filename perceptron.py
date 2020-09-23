# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import util
import time
PRINT = True

class PerceptronClassifier:
	"""
	Perceptron classifier.
	
	Note that the variable 'datum' in this code refers to a counter of features
	(not to a raw samples.Datum).
	"""
	def __init__( self, legalLabels, max_iterations):
		self.legalLabels = legalLabels
		self.type = "perceptron"
		self.max_iterations = max_iterations
		self.weights = {}
		for label in legalLabels:
			self.weights[label] = util.Counter() # this is the data-structure you should use

	def setWeights(self, weights):
		assert len(weights) == len(self.legalLabels);
		self.weights = weights;
			
	def train( self, trainingData, trainingLabels, validationData, validationLabels ):
		"""
		The training loop for the perceptron passes through the training data several
		times and updates the weight vector for each label based on classification errors.
		See the project description for details. 
		
		Use the provided self.weights[label] data structure so that 
		the classify method works correctly. Also, recall that a
		datum is a counter from features to values for those features
		(and thus represents a vector a values).
		"""
		start_time = time.time()
		self.features = trainingData[0].keys()
		for iteration in range(self.max_iterations): #Run the algorithm for a certain number of iterations
			print "Starting iteration ", iteration, "..."
			for i in range(len(trainingData)): #Have to run the algorithm on every datum in the training set
				datum = trainingData[i]
				correctNum = trainingLabels[i]
				bestWeight = 0
				bestNum = 0
				for x in range(len(self.legalLabels)): #Have to find the best weights among all of the legal labels (0,1) for faces (0...9) for digits
					weight = 0
					for k in range(len(self.features)): #Calculate the dot product between the datum and the weights 
						weight = weight + (self.weights[x][self.features[k]] * datum[self.features[k]])
					if weight > bestWeight: #If the weight we just calculated is greater than our previous best weight, then we update bestWeight and bestNum to represent that label as our guess
						bestWeight = weight
						bestNum = x
				if correctNum != bestNum: #If our guess is incorrect, then we update the weights by subtracting the datum from our guess's weight and adding the datum to the correct label's weight
					for k in range(len(self.features)):
						self.weights[correctNum][self.features[k]] = self.weights[correctNum][self.features[k]] + datum[self.features[k]]
						self.weights[bestNum][self.features[k]] = self.weights[bestNum][self.features[k]] - datum[self.features[k]]
		print("Took %s seconds to train" % (time.time() - start_time))
		
	def classify(self, data ):
		"""
		Classifies each datum as the label that most closely matches the prototype vector
		for that label.	See the project description for details.
		
		Recall that a datum is a util.counter... 
		"""
		guesses = []
		for datum in data:
			vectors = util.Counter()
			for l in self.legalLabels:
				vectors[l] = self.weights[l] * datum
			guesses.append(vectors.argMax())
		return guesses
