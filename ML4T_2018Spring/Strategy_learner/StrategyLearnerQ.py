"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import random
from QLearner import QLearner

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
        self.states = None
        num_states = 500
        num_actions = 3
        alpha = 0.5 # 0.2
        gamma = 0.75 # 0.8
        rar = 0.65 # 0.5
        radr = 0.99
        dyna = 0
        verbose = False
        qleaner = QLearner(num_states, num_actions, alpha, gamma, rar, radr, dyna, verbose)
        self.learner = qleaner

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

        # prices = prices_all[syms]  # only portfolio symbols
        # prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        # if self.verbose: print prices



        # Create an NxD dateframe holding all features for all dates

        data = prices_all.copy()
        data['volume'] = ut.get_data(syms, dates, colname="Volume", addSPY=False)
        # data['price'] = prices[self.symbol]
        # data['price'] = data['price'] / data['price'][0]
        data = self.clean_features(data)
        self.compute_stateBuckets(data)
        self.define_states(data)

        # Begin Training

        counter = 0
        # converged = True
        while counter < TRAIN_COUNT:

            date = 0
            # total_reward = 0
            data['position'] = 0
            pos = USD

            #Run on first day
            state = self.states[date]
            action = self.learner.querysetstate(int(state))


            # Iterate through all days
            for date in range(1, data.shape[0]):


                if self.assets >= 1000:
                    pos = LONG
                elif self.assets <= -1000:
                    pos = SHORT
                else:
                    pos = USD


                # if (self.verbose):
                #     print "Day " + str(date) + ": State = " + str(state) + " |  Position = " + str(data['position'][date - 1])
                reward = 0

                if action == HOLD:
                    reward += -0.5
                    if (pos == LONG):
                        reward += self.assets * data.ix[date, 'daily_ret']
                        # reward += self.days * (self.lastPrice / data.ix[date, symbol])
                        # self.days = self.days + 1
                    if (pos == SHORT):
                        reward += self.assets * data.ix[date, 'daily_ret']
                        # reward += self.days * (self.lastPrice / data.ix[date, symbol])
                        # self.days = self.days + 1

                ret = self.lastPrice - data.ix[date, symbol]

                bought = 0
                sold = 0


                # if action == BUY and pos != LONG:
                #     cost = 1000 * data.ix[date, self.symbol]
                #     self.assets = self.assets + 1000
                #     self.USD = self.USD - cost
                #
                #     ret = self.lastPrice - data.ix[date, symbol]
                #     bought = 1
                #
                # if action == SELL and pos != SHORT:
                #     cost = 1000 * data.ix[date, self.symbol]
                #     self.assets = self.assets - 1000
                #     self.USD = self.USD + cost
                #
                #     ret = self.lastPrice - data.ix[date, symbol]
                #     sold = 0

                if action == BUY:
                    if pos == SHORT:
                        self.assets = self.assets + 2000
                        ret = ret * 2
                        bought = 1

                    elif pos == USD:
                        # cost = 1000 * prices_all.ix[date, self.symbol]
                        self.assets = self.assets + 1000
                        # self.USD = self.USD - cost
                        bought = 1

                if action == SELL:
                    if pos == LONG:
                        self.assets = self.assets - 2000
                        ret = ret * 2
                        sold = 1

                    elif pos == USD:
                        # cost = 1000 * prices_all.ix[date, self.symbol]
                        self.assets = self.assets - 1000
                        # self.USD = self.USD + cost
                        sold = 1

                if bought == 1:
                    reward =  ret
                    self.lastPrice = data.ix[date, symbol]
                    self.days = 0
                elif sold == 1:
                    reward = ret * -1
                    self.lastPrice = data.ix[date, symbol]
                    self.days = 0

                # else:
                #     ret = ret + self.days * (self.lastPrice / data.ix[date, symbol])
                #     reward = ret
                #     self.days = self.days + 1



                state = self.states[date]
                action = self.learner.query(int(state), reward)

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
        self.stateBuckets = np.zeros(shape=(STATE_BUCKETS, 5))

        means_df = data['MA7'].copy()
        bbr_df = data['BBRatio'].copy()
        # price_df = data['price'].copy()
        # dr_df = data['daily_ret'].copy()
        mv7_df = data['MV7'].copy()


        means_df.sort_values(inplace=True)
        bbr_df.sort_values(inplace=True)
        # price_df.sort_values(inplace=True)
        # dr_df.sort_values(inplace=True)
        mv7_df.sort_values(inplace=True)

        bucketsize = data.shape[0]/STATE_BUCKETS
        for i in range(STATE_BUCKETS):
            # print bucketsize
            bracket = ((i+1) * bucketsize)-1
            # print bracket

            self.stateBuckets[i, 0] = means_df[bracket]
            self.stateBuckets[i, 1] = bbr_df[bracket]
            # self.stateBuckets[i, 2] = price_df[bracket]
            # self.stateBuckets[i, 3] = dr_df[bracket]
            self.stateBuckets[i, 4] = mv7_df[bracket]

        # print self.stateBuckets


    def define_states(self, data):
        df = data.copy()
        df['states'] = 0
        df['MA_disc'] = 0
        df['BBR_disc'] = 0
        df['MV7_disc'] = 0
        df['DR_disc'] = 0
        df['Price_disc'] = 0
        #Define the state for each date
        for date in range(0, data.shape[0]):
            for bucketNum in range(STATE_BUCKETS):
                if data.ix[date, 'MA7'] <= self.stateBuckets[bucketNum, 0]:
                    df.ix[date, 'MA_disc'] += bucketNum
                    break

            for bucketNum in range(STATE_BUCKETS):
                if data.ix[date, 'BBRatio'] <= self.stateBuckets[bucketNum, 1]:
                    df.ix[date, 'BBR_disc'] += bucketNum
                    break

            # for bucketNum in range(STATE_BUCKETS):
            #     if data.ix[date, 'daily_ret'] <= self.stateBuckets[bucketNum, 3]:
            #         df.ix[date, 'DR_disc'] += bucketNum
            #         break

            # for bucketNum in range(STATE_BUCKETS):
            #     if data.ix[date, 'price'] <= self.stateBuckets[bucketNum, 2]:
            #         df.ix[date, 'Price_disc'] += bucketNum
            #         break

            for bucketNum in range(STATE_BUCKETS):
                if data.ix[date, 'MV7'] <= self.stateBuckets[bucketNum, 4]:
                    df.ix[date, 'MV7_disc'] += bucketNum
                    break

        df['states'] = df['MA_disc']*100 + df['BBR_disc']*10 + df['MV7_disc']
        # print df['states']
        self.states = df['states']

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

        state = self.states[0]
        action = self.learner.querysetstate(int(state))
        # if self.verbose:
        #     date = 0
        #     tradesData.ix[date, 'USD'] = self.USD
        #     tradesData.ix[date, 'assets'] = self.assets
        #     tradesData.ix[date, 'state'] = state
        #     tradesData.ix[date, 'action'] = action
        #     tradesData.ix[date, 'pos'] = USD

        for date in range(1, prices_all.shape[0]):

            if self.assets == 1000:
                pos = LONG
            elif self.assets == -1000:
                pos = SHORT
            else:
                pos = USD

            # if self.verbose:
            #     tradesData.ix[date, 'USD'] = self.USD
            #     tradesData.ix[date, 'assets'] = self.assets
            #     tradesData.ix[date, 'state'] = state
            #     tradesData.ix[date, 'action'] = action
            #     tradesData.ix[date, 'pos'] = pos

            if action == BUY:
                if pos == SHORT:
                    trades[symbol][date] = 2000
                    self.assets = self.assets + 2000

                elif pos == USD:
                    trades[symbol][date] = 1000
                    # cost = 1000 * prices_all.ix[date, self.symbol]
                    self.assets = self.assets + 1000
                    # self.USD = self.USD - cost

            if action == SELL:
                if pos == LONG:
                    trades[symbol][date] = -2000
                    self.assets = self.assets - 2000

                elif pos == USD:
                    trades[symbol][date] = -1000
                    # cost = 1000 * prices_all.ix[date, self.symbol]
                    self.assets = self.assets - 1000
                    # self.USD = self.USD + cost

            if action == HOLD:
                trades[symbol][date] = 0

            # if self.verbose:
            #     tradesData.ix[date, 'trade'] = trades[symbol][date]

            state = self.states[date]
            action = self.learner.querysetstate(int(state))

        # print "Assets held: " + str(self.assets)
        # print "USD Held: " + str(self.USD)
        # print tradesData
        # print trades

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
    # st.testPolicy()
