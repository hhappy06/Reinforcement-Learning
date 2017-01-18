# ouput a policy for agent

import gym
import sys
import random
import numpy as np

# initialize random number seed
random.seed(0)

# define policy class
class Policy:
	def __init__(self, env):
		nAction = env.action_space.n
		shapeOb = env.observation_space.shape
		nFeature = nAction*shapeOb[0];

		self.actions = [i for i in xrange(nAction)]
		self.theta = [0.0 for i in xrange(nFeature)]
		self.theta = np.array(self.theta)
		self.theta = np.transpose(self.theta)

		# avoid implicit return
		return

	def get_feature_from_state_action(self, state, action):
		# initialize feature vector
		feature = np.array([0.0 for i in xrange(len(self.theta))])

		# one hot coded using state and action
		idx = 0
		for i in xrange(len(self.actions)):
			if action == self.actions[i]:
				idx = i
				break
		offset = idx * len(state)

		for i in xrange(len(state)):
			feature[i+offset] = state[i]

		return feature

	def get_derivative(self, state, action):
		return self.get_feature_from_state_action(state, action)

	# calculate Q(s, a)
	def get_Qfunction(self, state, action):
		# calcualte feature from state and action
		feature = self.get_feature_from_state_action(state, action)
		return np.dot(feature, self.theta)

	# get an action according to the policy
	# now only supprot greedy and epsilon-greedy
	def get_action_according_policy(self, state, policy_method = 'greedy', epsilon = 0.2):
		nAction = len(self.actions)
		qvalue = [0.0 for i in xrange(nAction)]

		maxQ = 0.0
		maxIdx = 0
		for i in xrange(nAction):
			qvalue[i] = self.get_Qfunction(state, self.actions[i])
			if maxQ < qvalue[i]:
				maxQ = qvalue[i]
				maxIdx = i

		if policy_method == 'greedy':
			return self.actions[maxIdx]

		# calculate action probality according to its Qvalue
		probality = [0.0 for i in xrange(nAction)]
		probality[maxIdx] = 1.0 - epsilon;
		for i in xrange(nAction):
			probality[i] += epsilon/nAction;

		random_number = random.random()
		sumPro = 0.0
		for i in xrange(nAction):
			sumPro += probality[i]
			if sumPro > random_number :
				return self.actions[i];

		# default action 
		return maxIdx
