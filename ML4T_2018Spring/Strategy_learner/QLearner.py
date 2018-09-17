"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 200, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.qtable = np.zeros(shape=(num_states, num_actions))
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.T = []
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        self.s = s
        #action = np.argmax(self.qtable[s, :], axis=0)
        action = rand.randint(0, self.num_actions-1)
        if rand.random > self.rar:
            action = np.argmax(self.qtable[s, :], axis=0)
        self.a = action

        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: real valued immediate reward
        @returns: The selected action
        """
        action = rand.randint(0, self.num_actions-1)
        value = self.qtable[self.s, self.a] + self.alpha * (r + (self.gamma * self.qtable[s_prime, np.argmax(self.qtable[s_prime,:])]) - self.qtable[self.s, self.a])
        self.qtable[self.s, self.a] = value
        self.T.append((self.s, self.a, s_prime, r))

        if self.dyna != 0:
            index = np.random.choice(len(self.T), size=self.dyna, replace=True)
            for i in index:
                state, action, s_p, r = self.T[i]
                value = self.qtable[state, action] + self.alpha * (
                            r + (self.gamma * self.qtable[s_p, np.argmax(self.qtable[s_p, :])]) - self.qtable[
                        state, action])
                self.qtable[state, action] = value

        if rand.random > self.rar:
            action = np.argmax(self.qtable[s_prime, :])

        self.rar = self.rar * self.radr
        self.s = s_prime
        self.a = action



        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

    def author(self):
        return 'gsharpe9'

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
