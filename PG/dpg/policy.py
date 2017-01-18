# output a policy for agent

import gym
import sys
import random
import numpy as np

# initialize random number seed
random.seed(0)

# define policy class
# state = [env.state,1], we extend the state's dimension to add a bias
class Policy:
	def __init__(self, env):
		# state dimension: N
		# action dimension: M
		# dpg function: [Nx M]
		self.state_dim = env.observation_space.shape[0]
		self.action_dim = len(env.action_space.sample())
		
		# normalize state space
		ob_examples = np.array([env.observation_space.sample() for _ in xrange(10000)])
		self.state_mean = np.mean(ob_examples, axis = 0)
		self.state_std = np.std(ob_examples, axis= 0) + 1.0e-6

		# initialize parameters for DPG
        # \mu_{s} = theta.T * state ()
		self.theta_policy = (np.random.rand(self.state_dim, self.action_dim) - 0.5)* 0.03
		# self.theta_policy[0][0] = 0.1
		# self.theta_policy[1][0] = 10
		# q(s,a) = w.T * (delt(\mu(s))) * (a-\mu(s)) + theta_value.T * s
		self.weight_qfunction = (np.random.rand(self.state_dim * self.action_dim)-0.5) * 0.03
		# V(s) = theta_value.T * s
		self.theta_value = (np.random.rand(self.state_dim)-0.5) * 0.03
		
		self.sigma = 0.8
		self.sigma_decay = 0.5
		self.sigma_decay_step = 4000
		self.sigma_count = 0
		
		# avoid implicit return
		return

	def normalize_state(self, state):
		return (state - self.state_mean) / self.state_std
		
	# get an action according to the policy
	# now only support probability from Gaussian Distribution
	def get_action_from_policy(self, state, is_explore = False):
		state = self.normalize_state(state)
		action_mean = np.tanh(np.dot(self.theta_policy.T, state))
		if not is_explore:
			return action_mean
		
		cov_mat = np.eye(self.action_dim)*self.sigma
		action = np.random.multivariate_normal(action_mean, cov_mat)
		return np.clip(action, -1, 1)
	
	def get_dev_theta_policy(self, nor_state):
		# return Jacobi matrix of dpg
		action_mean = np.tanh(np.dot(self.theta_policy.T, nor_state))
		dev_theta_policy = np.zeros([self.state_dim * self.action_dim, self.action_dim])
		for i in range(self.theta_policy.shape[1]):
			dev_theta_policy[:,i][i*self.state_dim : (i+1)*self.state_dim] = nor_state * (1.0 - action_mean[i] * action_mean[i])
		return dev_theta_policy
	
	def get_dev_theta_value(self, nor_state):
		return nor_state

	def get_Q_value(self, nor_state, action):
		action_mu = np.tanh(np.dot(self.theta_policy.T, nor_state))
		dev_theta_policy = self.get_dev_theta_policy(nor_state)
		qvalue = np.dot(self.weight_qfunction.T, dev_theta_policy)
		qvalue = np.dot(qvalue, (action - action_mu)) + np.dot(self.theta_value.T, nor_state)
		return qvalue
			
	def update_policy_batch(self, batch, gamma, lr_policy, lr_qfunction, lr_qvalue):
		nbatch = len(batch)
		if nbatch == 0:
			return 0

		# update policy
		sum_dev_theta_policy = np.zeros(self.theta_policy.shape)
		sum_dev_weight_qfunction = np.zeros(self.weight_qfunction.shape)
		sum_dev_theta_value = np.zeros(self.theta_value.shape)
		
		sum_q_hat = 0.0
		sum_q_hat2 = 0.0
		for item in batch:
			# normalize state
			item0 = self.normalize_state(item[0])
			item3 = self.normalize_state(item[3])
			q_hat = item[2]
			if not item[5]:
				action_mu = self.get_action_from_policy(item[3], False)
				q_hat += gamma*self.get_Q_value(item3, action_mu)
				if np.isnan(q_hat):
					exit()
				
			sum_q_hat += item[2]
			sum_q_hat2 += q_hat
			q = self.get_Q_value(item0, item[1])
			delt = q_hat - q
			
			jacobi_theta_policy = self.get_dev_theta_policy(item0)
			
			dev_theta_policy = np.dot(self.weight_qfunction.T, jacobi_theta_policy)
			dev_theta_policy = np.dot(jacobi_theta_policy, dev_theta_policy).reshape([self.action_dim, self.state_dim]).T
			sum_dev_theta_policy +=  dev_theta_policy
				
			action0 = self.get_action_from_policy(item[0], False)
			dev_weight_qfunction = np.dot(jacobi_theta_policy, (item[1] - action0))
			sum_dev_weight_qfunction += delt * dev_weight_qfunction
			
			dev_theta_value = self.get_dev_theta_value(item0)
			sum_dev_theta_value += delt * dev_theta_value

		self.theta_policy += lr_policy * sum_dev_theta_policy/nbatch
		self.weight_qfunction += lr_qfunction * sum_dev_weight_qfunction /nbatch
		self.theta_value += lr_qvalue * sum_dev_theta_value / nbatch
		
		print 'theta:{0},{1}'.format(self.theta_policy[0], self.theta_policy[1])
		# update decay
		self.sigma_count = (self.sigma_count + 1) % self.sigma_decay_step
		if self.sigma_count == 0 and self.sigma > 0.02:
			self.sigma *= self.sigma_decay

		return sum_q_hat / nbatch, sum_q_hat2 / nbatch
