import pandas as pd
import numpy as np
from collections import Counter
import random
from scipy.stats import mode


class RTLearner:

    def __init__(self, leaf_size = 5, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.table = None

    def addEvidence(self, Xtrain, Ytrain):
        newtable = self.buildTree(Xtrain, Ytrain)

        if self.table is None:
            self.table = newtable
        else:
            self.table = np.vstack((self.table, newtable))

    def buildTree(self, Xtrain, Ytrain, Yroot=[]):
        numSamples = Xtrain.shape[0]

        if numSamples == 0:
            mostCommom = Counter(Yroot).most_common(1)[0][0]
            return np.array([-1, mostCommom, np.nan, np.nan])
        if numSamples <= self.leaf_size:
            #mostCommom = np.mean(Ytrain)#mode(Ytrain, axis=0)
            mostCommom = mode(Ytrain)
            ind = np.argmax(mostCommom[0])
            mostCommom = mostCommom[1][ind]
            arr = np.asarray([[-1, mostCommom, np.nan, np.nan]])
            #arr = arr[np.newaxis]
            #print arr
            return arr
        #y = len(set(Ytrain[0]))
        #print Ytrain[0][0]
        num = len(np.unique(Ytrain))
        #print num
        if num == 1:
            arr = np.asarray([[-1, np.mean(Ytrain), np.nan, np.nan]])
            #arr = arr[np.newaxis]
            #print arr
            return arr
        # if num == 1 or numSamples <= self.leaf_size:
        #     #mostCommom = Counter(Ytrain).most_common(1)[0][0]
        #     mostCommom = mode(Ytrain, axis=0)
        #     #print mostCommom[0][0][0]
        #     print np.array([-1, mostCommom[0][0][0], np.nan, np.nan])[np.newaxis]
        #
        #     return np.array([-1, mostCommom[0][0][0], np.nan, np.nan])[np.newaxis]

        numFeats = Xtrain.shape[1]


        #Setup feature  list to randomly pick from
        featList = list(range(numFeats))

        best, splitVal, left, right, addleaf = self.selectRandomFeature(Xtrain, featList)
        if addleaf:
            mostCommom = mode(Ytrain)
            ind =np.argmax(mostCommom[0])
            mostCommom = mostCommom[1][ind]
            return np.asarray([[-1, mostCommom, np.nan, np.nan]])

        left = self.buildTree(Xtrain[left], Ytrain[left], Ytrain)
        right = self.buildTree(Xtrain[right], Ytrain[right], Ytrain)

        # starting row for the right subtree
        if left.ndim == 1:
            rightStartRow = 2
        else:
            rightStartRow = left.shape[0] + 1

        root = np.array([best, splitVal, 1, rightStartRow])
        return np.vstack((root, left, right))

    def selectRandomFeature(self, Xtrain, featList):
        """
        :return: Best feature split by correlation
        """
        counter = 0
        while len(featList) > 0:
            best = random.choice(featList)
            median = np.median(Xtrain[:, best])
            left = Xtrain[:, best] <= median
            right = Xtrain[:, best] > median

            if len(np.unique(left)) != 1:
                break

            featList.remove(best)
            counter += 1

        addLeaf = False
        if len(featList) == 0:
            addLeaf = True

        return best, median, left, right, addLeaf

    def query(self, Xtests):
        yhats = []
        for test in Xtests:
            yhats.append(self.treeSearch(test, 0))
        return np.asarray(yhats)

    def treeSearch(self, test, DTrow):

        feature, split = self.table[DTrow, 0:2]

        if feature == -1:
            return split
        elif test[int(feature)] <= split:
            yhat = self.treeSearch(test, DTrow + int(self.table[DTrow, 2]))
        else:
            yhat = self.treeSearch(test, DTrow + int(self.table[DTrow, 3]))

        return yhat

    def author(self):
        return 'gsharpe9' # replace tb34 with your Georgia Tech username.



