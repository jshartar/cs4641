import numpy as np
import random

class BagLearner:

    def __init__(self, learner = None, kwargs = {"argument1":1, "argument2":2}, bags = 20, boost = False, verbose = False):
        self.bagsNum = bags
        self.boost = boost
        self.verbose = verbose
        learners = []
        # kwargs.update({"verbose":verbose, "leaf_size":kwargs.get("arguement1")})
        for i in range(0, bags):
            learners.append(learner(**kwargs))
        self.learners = learners
        self.kwargs = kwargs
        self.bagsX = None
        self.bagsY = None

    def addEvidence(self, Xtrain, Ytrain):
        numSamples = Xtrain.shape[0]
        # bags = list((list(), list()))
        bagsX = np.empty([Xtrain.shape[0], Xtrain.shape[1]])
        bagsY = np.empty([Ytrain.shape[0],1])

        for n in range(numSamples):
            #print Xtrain
            ran = random.randrange(0, numSamples)
            item = Xtrain.ix[int(ran)]
            np.append(bagsX, item)
            #print bagsX
            #bagsY.ix[n] = Ytrain.ix[int(ran)]
            np.append(bagsY, Ytrain.ix[int(ran)])
            bagsY = np.asarray(bagsY)
            #print bagsY

        # self.bags = bags

        bagsX = np.asarray(bagsX)
        bagsY = np.asarray(bagsY)
        self.bagsX = bagsX
        self.bagsY = bagsY

        for learner in self.learners:
            learner.addEvidence(bagsX, bagsY)


    def query(self, Xtests):
        for bag in range(self.bagsNum):
            bagvotes = []
            for learner in self.learners:
                bagvotes.append(learner.query(Xtests))
            votes = np.asarray(bagvotes).transpose()

        votes = np.mean(votes, axis=1)

        return votes


    def author(self):
        return 'gsharpe9' # replace tb34 with your Georgia Tech username.