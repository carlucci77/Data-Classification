# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math
import time

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.
    
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
        self.prior = util.Counter()
        self.datumProb = util.Counter()
        
    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """    
            
        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
        
        if (self.automaticTuning):
                kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
                kgrid = [self.k]
                
        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
            
    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter 
        that gives the best accuracy on the held-out validationData.
        
        trainingData and validationData are lists of feature Counters. The corresponding
        label lists contain the correct label for each datum.
        
        To get the list of all possible features or labels, use self.features and 
        self.legalLabels.
        """
        counter = 0
        k = 0.0
        for x in range(len(self.legalLabels)): #Choose different smoothing value if working on faces or digits
            counter = counter + 1
        if counter > 2:
            k = 0.001
        else:
            k = 0.5
        start_time = time.time()
        for x in range(len(trainingLabels)): #Go through all the possible labels and count each instance of a certain label. We will use these counts to calculate the prior probability
            num = trainingLabels[x]
            self.prior[num] = self.prior[num] + 1
        for x in range(len(trainingData)):
            datum = trainingData[x]
            num = trainingLabels[x]
            for key, value in datum.items():
                if self.datumProb[key,num] == 0:
                    self.datumProb[key,num] = [0,0,0] #Add array to each counter index to hold the three probabilities of the values
                if value == 2:
                    self.datumProb[key, num][2] = 1 + self.datumProb[key, num][2] #Add together all the times that this exact pixel is this value for the given label
                elif value == 1:
                    self.datumProb[key,num][1] = 1 + self.datumProb[key, num][1]
                else:
                    self.datumProb[key,num][0] = 1 + self.datumProb[key, num][0]
        for x, value in self.datumProb.items(): #Divide all of the pixel values by the total occurrence of that label and add k to numerator and denominator for smoothing
            self.datumProb[x][0] = float(value[0] + k)/float(self.prior[x[1]] + k)
            self.datumProb[x][1] = float(value[1] + k)/float(self.prior[x[1]] + k)
            self.datumProb[x][2] = float(value[2] + k)/float(self.prior[x[1]] + k)
        for num in range(len(self.legalLabels)): #Divide each occurrence count of the labels by the total amount of labels to get the prior probability for each label
            self.prior[num] = float(self.prior[num])/float(len(trainingLabels))
        print("Took %s seconds to train" % (time.time() - start_time))
        
    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        
        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses
            
    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.        
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
        
        To get the list of all possible features or labels, use self.features and 
        self.legalLabels.
        """
        logJoint = util.Counter()
        for x in range(len(self.legalLabels)):
            num = self.legalLabels[x]
            logJoint[num] = math.log(self.prior[num]) #For every possible label, for its logJoint counter index, we are adding the prior probability for P(y) for the corresponding label
            for key, value in datum.items(): #We traverse through every pixel in the datum
                if value == 0:
                    if self.datumProb[key,num][0] != 0: #Ensure the probability is not 0 since we cannot take the log of 0
                        logJoint[num] = logJoint[num] + math.log(self.datumProb[key,num][0]) #Add the probability of that pixel being that feature value given a certain label to the logJoint index for the corresponding label
                elif value == 1:
                    if self.datumProb[key,num][1] != 0:
                        logJoint[num] = logJoint[num] + math.log(self.datumProb[key,num][1])
                else:
                    if self.datumProb[key,num][2] != 0:
                        logJoint[num] = logJoint[num] + math.log(self.datumProb[key,num][2])
        return logJoint
    