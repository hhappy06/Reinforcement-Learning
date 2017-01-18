import gym
import numpy as np
import tensorflow as tf
import math
from Actor import Actor
from Critic import Critic

GAMMA = 0.99

SAVE_PATH = 'networks/'

class ActorCritic:
	def __init__(self, state_dim, action_dim):
		self.name = 'ActorCritic'
		self.state_dim = state_dim
		self.action_dim = action_dim
		
		self.time_update = 0
		self.sess = tf.InteractiveSession()
		
		#initialize actor and critic network
		self.actor = Actor(self.sess, state_dim, action_dim)
		self.critic = Critic(self.sess, state_dim, action_dim)
		
		# explore parameter
		self.sigma = 2
		self.sigma_decay = 0.5
		self.sigma_min = 0.1
		self.sigma_decay_step = 50000
		self.sigma_count = 0
		
		# save network
		self.saver = tf.train.Saver()
				
	def train(self, batch):
		if(len(batch)) == 0:
			return
		self.time_update += 1
		state0_batch = np.asarray([data[0] for data in batch])
		action0_batch = np.asarray([data[1] for data in batch])
		reward1_batch = np.asarray([data[2] for data in batch])
		state1_batch = np.asarray([data[3] for data in batch])
		isTerminal_batch = np.asarray([data[4] for data in batch])
		
		action0_batch = np.resize(action0_batch, [len(batch), self.action_dim])
		action1_batch = self.actor.calc_target_actions(state1_batch)
		q_value_batch = self.critic.calc_target_q(state1_batch, action1_batch)
		
		target_reward = []
		for idx in range(len(batch)):
			if not isTerminal_batch[idx]:
				target_reward.append(reward1_batch[idx] + GAMMA * q_value_batch[idx])
			else:
				target_reward.append(reward1_batch[idx])
	
		target_reward = np.resize(target_reward, [len(batch), 1])
		
		# update critic network
		self.critic.train_q(target_reward, state0_batch, action0_batch)
		
		# update policy using Q-gradient
		action_for_gradient = self.actor.calc_actions(state0_batch)
		q_gradient_batch = self.critic.calc_gradients(state0_batch, action_for_gradient)
		
		self.actor.train_p(q_gradient_batch, state0_batch)
		
		# update target network
		self.actor.do_target_update()
		self.critic.do_target_update()
		
		return np.mean(reward1_batch), np.mean(target_reward), self.sigma
	
	def action(self, state):
		return self.actor.calc_action(state)
	
	def action_explore(self, state):
		action = self.actor.calc_action(state)
		cov_mat = np.eye(self.action_dim) * self.sigma
		action = np.random.multivariate_normal(action, cov_mat)
		
		# update explore sigma
		self.sigma_count = (self.sigma_count + 1) % self.sigma_decay_step
		if self.sigma > self.sigma_min and self.sigma_count == 0:
			self.sigma *= self.sigma_decay
		
		action = np.clip(action, -1, 1)
		return action
	
	def save_network(self, file_name = '_network'):
		self.saver.save(self.sess, SAVE_PATH + file_name, global_step = self.time_update)
	
	def load_network(self, file_name = '_network'):
		self.saver.restore(self.sess, SAVE_PATH + file_name)