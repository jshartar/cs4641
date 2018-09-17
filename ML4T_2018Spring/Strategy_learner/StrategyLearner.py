"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import random
#from QLearner import QLearner
from BagLearner import BagLearner
from RTLearner import RTLearner

# Possible Actions
HOLD = 0
BUY = 1
SELL = 2

#Possible positions
USD = 0
LONG = 1
SHORT = -1

TRAIN_COUNT = 1
STATE_BUCKETS = 5



class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.symbol = None
        self.Ytrain = None
        num_states = 500
        num_actions = 3

        leafSize = 20
        verbose = False
        baglearner = BagLearner(RTLearner, kwargs = {"leaf_size":20, "verbose":False}, bags = 10, boost = False, verbose=False)
        #qleaner = QLearner(num_states, num_actions, alpha, gamma, rar, radr, dyna, verbose)
        self.learner = baglearner

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 100000):

        self.symbol = symbol
        self.USD = sv
        self.assets = 0
        self.lastPrice = 0
        self.days = 0

        # add your code to do learning here
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates, addSPY=False)  # automatically adds SPY

        prices = prices_all[syms]  # only portfolio symbols
        #prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        #if self.verbose: print prices



        # Create an NxD dateframe holding all features for all dates

        data = prices_all.copy()
        data['volume'] = ut.get_data(syms, dates, colname="Volume", addSPY=False)
        data['price'] = prices[self.symbol]
        data['price'] = data['price'] / data['price'][0]


        x_train = self.clean_features(data)
        self.compute_stateBuckets(data)
        self.define_states(data)
        # x_train = data


        # Begin Training

        counter = 0
        # converged = True
        while counter < TRAIN_COUNT:

            # Iterate through all days
            x_train.fillna(method='ffill', inplace=True)
            x_train.fillna(method='bfill', inplace=True)
            x_train.reset_index(drop=True, inplace=True)
            self.Ytrain.reset_index(drop=True, inplace=True)

            #print self.Ytrain.shape
            #print self.Ytrain
            self.learner.addEvidence(x_train, self.Ytrain)

        counter += 1




    def clean_features(self, data):

        data['daily_ret'] = (data[self.symbol] / data[self.symbol].shift(1)) - 1
        # data[symbol] = data[symbol] / data[symbol][0]
        data['MA7'] = data[self.symbol].rolling(7).mean().round(3)
        data['LowerBB'] = data['MA7'] - 2 * data[self.symbol].rolling(7).std()
        data['UpperBB'] = data['MA7'] + 2 * data[self.symbol].rolling(7).std()
        data['BBRatio'] = (data[self.symbol] - data['LowerBB']) / (data['UpperBB'] - data['LowerBB'])
        del data['LowerBB']
        del data['UpperBB']

        data['MV7'] = data['volume'].rolling(7).mean().round(3)

        return data

    def compute_stateBuckets(self, data):
        pass


    def define_states(self, data):

        ytrain = data[[self.symbol]].copy()  # only portfolio symbols
        ytrain.values[:, :] = int(0)  # set them all to nothing

        for date in range(len(data)):
            if random.random < 0.5:
                ytrain.ix[date] = 1


        self.Ytrain = ytrain

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 100000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        # trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        trades.values[:,:] = int(0) # set them all to nothing

        data = prices_all.copy()
        data['volume'] = ut.get_data([symbol], dates, colname="Volume", addSPY=False)
        # data['price'] = trades[self.symbol]
        # data['price'] = data['price'] / data['price'][0]
        data = self.clean_features(data)
        self.compute_stateBuckets(data)
        self.define_states(data)

        # if self.verbose:
        #     tradesData = prices_all[[symbol, ]]  # only portfolio symbols
        #     tradesData.values[:, :] = 0  # set them all to nothing

        # self.USD = sv
        self.assets = 0

        # if self.verbose:
        #     date = 0
        #     tradesData.ix[date, 'USD'] = self.USD
        #     tradesData.ix[date, 'assets'] = self.assets
        #     tradesData.ix[date, 'state'] = state
        #     tradesData.ix[date, 'action'] = action
        #     tradesData.ix[date, 'pos'] = USD

        for date in range(1, prices_all.shape[0]):
            print data[date]
            self.learner.query(data[date])



        return trades




        ##################################################
        # trades.values[0,:] = 1000 # add a BUY at the start
        # trades.values[40,:] = -1000 # add a SELL
        # trades.values[41,:] = 1000 # add a BUY
        # trades.values[60,:] = -2000 # go short from long
        # trades.values[61,:] = 2000 # go long from short
        # trades.values[-1,:] = -1000 #exit on the last day
        # if self.verbose: print type(trades) # it better be a DataFrame!
        # if self.verbose: print trades
        # if self.verbose: print prices_all
        #################################################
        return trades

if __name__=="__main__":
    st = StrategyLearner(verbose=True)
    st.addEvidence()
    st.testPolicy()
