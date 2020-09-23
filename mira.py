# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util
import time
PRINT = True

class MiraClassifier:
	"""
	Mira classifier.
	
	Note that the variable 'datum' in this code refers to a counter of features
	(not to a raw samples.Datum).
	"""
	def __init__( self, legalLabels, max_iterations):
		self.legalLabels = legalLabels
		self.type = "mira"
		self.automaticTuning = False 
		self.C = 0.001
		self.legalLabels = legalLabels
		self.max_iterations = max_iterations
		self.initializeWeightsToZero()

	def initializeWeightsToZero(self):
		"Resets the weights of each label to zero vectors" 
		self.weights = {}
		for label in self.legalLabels:
			self.weights[label] = util.Counter() # this is the data-structure you should use
	
	def train(self, trainingData, trainingLabels, validationData, validationLabels):
		"Outside shell to call your method. Do not modify this method."	
			
		self.features = trainingData[0].keys() # this could be useful for your code later...
		
		if (self.automaticTuning):
				Cgrid = [0.002, 0.004, 0.008]
		else:
				Cgrid = [self.C]
				
		return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

	def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
		"""
		This method sets self.weights using MIRA.	Train the classifier for each value of C in Cgrid, 
		then store the weights that give the best accuracy on the validationData.
		
		Use the provided self.weights[label] data structure so that 
		the classify method works correctly. Also, recall that a
		datum is a counter from features to values for those features
		representing a vector of values.
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
					tau = 0 #But the datum must be multiplied by tau before being added
					for i in range(len(self.features)): #Multiplying the datum by itself
						tau = tau + (datum[self.features[i]] * datum[self.features[i]])
					tau = tau * 2 #Multiplying our result by 2
					temp1 = 0
					temp2 = 0
					for i in range(len(self.features)):
						temp1 = self.weights[bestNum][self.features[i]] * datum[self.features[i]]
						temp2 = self.weights[correctNum][self.features[i]] * datum[self.features[i]]
					weightDiff = (temp1 - temp2) + float(1) #Calculating (wy' - wy)f + 1
					tau = min(0.002,weightDiff/tau) #Choosing the min of these two as we do not want tau to become too big
					for k in range(len(self.features)): #Multiply datum by tau and then add or subtract it with the weight vector
						self.weights[correctNum][self.features[k]] = self.weights[correctNum][self.features[k]] + (datum[self.features[k]] * tau)
						self.weights[bestNum][self.features[k]] = self.weights[bestNum][self.features[k]] - (datum[self.features[k]] * tau)
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
