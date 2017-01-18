import numpy as np
import tensorflow as tf
import math

# Q(s, a) network layer
LAYER_STATE1 = 16
LAYER_STATE2 = 8

# learning rate
LR = 1.0e-2
LR_TAU = 1.0e-3
REGULARIZATION_FACTOR = 1.0e-4


class Actor:
	def __init__(self, sess, state_dim, action_dim):
		self.update_time = 0
		self.sess = sess
		
		# create a Q-function network
		self.state_input = tf.placeholder('float', [None, state_dim])
		
		var = 1 / math.sqrt(state_dim)
		state_w1 = tf.Variable(tf.random_uniform([state_dim, LAYER_STATE1], -var, var))
		state_b1 = tf.Variable(tf.random_uniform([LAYER_STATE1], -0.001, 0.001))
		state_layer1 = tf.nn.relu(tf.matmul(self.state_input, state_w1) + state_b1)
		
		var = 1 / math.sqrt(LAYER_STATE1)
		state_w2 = tf.Variable(tf.random_uniform([LAYER_STATE1, LAYER_STATE2], -var, var))
		state_b2 = tf.Variable(tf.random_uniform([LAYER_STATE2], -0.001, 0.001))
		state_layer2 = tf.nn.relu(tf.matmul(state_layer1, state_w2) + state_b2)
		
		
		var = 1 / math.sqrt(LAYER_STATE2)
		state_w3 = tf.Variable(tf.random_uniform([LAYER_STATE2, action_dim], -var, var))
		state_b3 = tf.Variable(tf.random_uniform([action_dim], -0.001, 0.001))
		self.action_output= tf.nn.tanh(tf.matmul(state_layer2, state_w3) + state_b3)
		
		self.net = [state_w1, state_b1, state_w2, state_b2, state_w3, state_b3]
		
		# define target q network
		self.target_state_input = tf.placeholder('float', [None, state_dim])
		
		ema = tf.train.ExponentialMovingAverage(decay=1 - LR_TAU)
		self.target_update = ema.apply(self.net)
		
		self.target_net = [ema.average(x) for x in self.net]
		
		target_layer1 = tf.nn.relu(tf.matmul(self.target_state_input, self.target_net[0]) + self.target_net[1])
		target_layer2 = tf.nn.relu(tf.matmul(target_layer1, self.target_net[2]) + self.target_net[3])
		self.target_action_output = tf.nn.tanh(tf.matmul(target_layer2, self.target_net[4]) + self.target_net[5])
		
		#define optimization
		self.q_gradient_input = tf.placeholder('float', [None, action_dim])
		self.parameters_gradient = tf.gradients(self.action_output, self.net, -self.q_gradient_input)
		self.opt = tf.train.AdamOptimizer(LR).apply_gradients(zip(self.parameters_gradient, self.net))
		
		# initialize network
		self.sess.run(tf.initialize_all_variables())
		self.do_target_update()
	
	def do_target_update(self):
		self.sess.run(self.target_update)
	
	def train_p(self, q_gradient_batch, state_batch):
		self.update_time += 1
		self.sess.run(self.opt, feed_dict={
			self.q_gradient_input: q_gradient_batch,
			self.state_input: state_batch
			})
	
	def calc_actions(self, state_batch):
		return self.sess.run(self.action_output, feed_dict={
			self.state_input: state_batch
			})
	
	def calc_action(self, state):
		return self.sess.run(self.action_output, feed_dict={
			self.state_input: [state]
			})[0]
	
	def calc_target_actions(self, state_batch):
		return self.sess.run(self.target_action_output, feed_dict={
			self.target_state_input: state_batch
			})

