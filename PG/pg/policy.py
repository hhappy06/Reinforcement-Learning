# ouput a policy for agent

import gym
import sys
import random
import numpy as np
import scipy as sp
from scipy.optimize import leastsq 

# initialize random number seed
random.seed(0)

# define policy class
# state = [env.state,1], we extend the state's dimension to add a bias
class Policy:
	def __init__(self, env):
		nAction = env.action_space.n
		shapeOb = env.observation_space.shape

		self.actions = np.array([i for i in xrange(nAction)])
		self.theta = np.random.rand(shapeOb[0],nAction)

		self.weight = np.random.rand(self.theta.size).T

		# avoid implicit return
		return

	def get_probability(self, state):
		res_exp = np.exp(np.dot(self.theta.T, state))
		sum_res_exp = np.sum(res_exp)
		probability = res_exp/sum_res_exp

		return probability

	# get an action according to the policy
	# now only supprot greedy and probability
	def get_action_according_policy(self, state):
		probability = self.get_probability(state)

		random_number = random.random()
		sumPro = 0.0
		for i in xrange(len(self.actions)):
			sumPro += probability[i]
			if sumPro > random_number :
				return self.actions[i];

		# default action 
		return self.actions[len(self.actions) - 1]

	def get_derivative(self, state, action):
		probability = self.get_probability(state)

		der_theta = np.zeros(self.theta.shape)

		action_idx = np.where(self.actions == action)[0][0]
		der_theta[:,action_idx] = state
		for i in xrange(len(self.actions)):
			der_theta[:,i] -= probability[i]*state

		return der_theta

	def update_policy_batch(self, batch, gamma, lr_policy, lr_qvalue):
		nbatch = len(batch)
		if nbatch == 0 :
			return 0

		# update policy
		sum_dev_policy = np.zeros(self.theta.shape)
		sum_dev_Qfunction = np.zeros(self.weight.shape) 

		sum_qhat = 0

		for item in batch:
			q_hat = item[2]
			if not item[5]:
				q_hat += gamma*self.get_qvalue(item[3],item[4])

			sum_qhat += q_hat
			dev0 = self.get_derivative(item[0],item[1])
			sum_dev_policy += dev0 * q_hat

			q = self.get_qvalue(item[0],item[1])
			sum_dev_Qfunction += (q - q_hat)*dev0.reshape(self.weight.size)

		self.theta += lr_policy * sum_dev_policy/nbatch
		self.weight -= sum_dev_Qfunction * lr_qvalue/nbatch

		# self.lsq_qvalue(batch, gamma, lr_qvalue)
 
		# print sum_qhat/nbatch

		# print self.sweight
		return sum_qhat/nbatch

	def get_qvalue(self, state, action):
		feature = self.get_derivative(state, action).reshape(self.theta.size).T	
		return np.dot(self.weight.T, feature)

	def lsq_qvalue(self, batch, gamma, lr_qvalue):
		nbatch = len(batch)
		if nbatch == 0 :
			return

		w0 = self.weight
		features = []
		qvalues = []

		for item in batch:
			q_hat = item[2]
			if not item[5]:
				q_hat += gamma*self.get_qvalue(item[3],item[4])
			f = self.get_derivative(item[0],item[1]).reshape(self.weight.size).T

			features.append(f.tolist())
			qvalues.append(q_hat)

		# w = np.linalg.lstsq(np.array(features), np.array(qvalues))
		w = np.linalg.lstsq(features, qvalues)

		self.weight += lr_qvalue*w[0]
