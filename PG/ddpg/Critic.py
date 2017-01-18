import numpy as np
import tensorflow as tf
import math

# Q(s, a) network layer
LAYER_STATE = 32
LAYER_ACTION = 16
LAYER_CONCAT = 16

# learning rate
LR = 5.0e-3
LR_TAU = 1.0e-3
REGULARIZATION_FACTOR = 1.0e-4

class Critic:
	def __init__(self, sess, state_dim, action_dim):
		self.update_time = 0
		self.sess = sess
		
		# create a Q-function network
		self.state_input = tf.placeholder('float', [None, state_dim])
		self.action_input = tf.placeholder('float', [None, action_dim])
		
		var = 1.0/math.sqrt(state_dim)
		state_w = tf.Variable(tf.random_uniform([state_dim, LAYER_STATE], -var, var))
		state_b = tf.Variable(tf.random_uniform([LAYER_STATE], -0.001, 0.001))
		state_layer = tf.nn.relu(tf.matmul(self.state_input, state_w) + state_b)
		
		action_w = tf.Variable(tf.random_uniform([action_dim, LAYER_ACTION], -var, var))
		action_b = tf.Variable(tf.random_uniform([LAYER_ACTION], -0.001, 0.001))
		action_layer = tf.nn.relu(tf.matmul(self.action_input, action_w) + action_b)

# 		concat variable
		state_action_cat = tf.concat(1, [state_layer, action_layer])
	
		var = 1.0/math.sqrt(LAYER_STATE + LAYER_ACTION)
		state_action_w = tf.Variable(tf.random_uniform([LAYER_STATE + LAYER_ACTION, LAYER_CONCAT], -var, var))
		state_action_b = tf.Variable(tf.random_uniform([LAYER_CONCAT], -0.001, 0.001))
		state_action_layer = tf.nn.relu(tf.matmul(state_action_cat, state_action_w) + state_action_b)
		
		var = 1.0 / math.sqrt(LAYER_CONCAT)
		output_layer_w = tf.Variable(tf.random_uniform([LAYER_CONCAT, 1], -var, var))
		output_layer_b = tf.Variable(tf.random_uniform([1], -0.001, 0.001))
		self.q_value_output = tf.identity(tf.matmul(state_action_layer, output_layer_w) + output_layer_b)
		
		self.net = [state_w, state_b, action_w, action_b, state_action_w, state_action_b, output_layer_w, output_layer_b]
		
# 		define target q network
		self.target_state_input = tf.placeholder('float', [None, state_dim])
		self.target_action_input = tf.placeholder('float', [None, action_dim])
		
		ema = tf.train.ExponentialMovingAverage(decay = 1 - LR_TAU)
		self.target_update = ema.apply(self.net)
		
		target_net = [ema.average(x) for x in self.net]
		
		layer1 = tf.nn.relu(tf.matmul(self.target_state_input, target_net[0]) + target_net[1])
		layer2 = tf.nn.relu(tf.matmul(self.target_action_input, target_net[2]) + target_net[3])
		cat_layer = tf.concat(1, [layer1, layer2])
		target_output_layer = tf.nn.relu(tf.matmul(cat_layer, target_net[4]) + target_net[5])
		self.target_q_value_output = tf.identity(tf.matmul(target_output_layer, target_net[6]) + target_net[7])
		
# 		define optimization
		self.q_input = tf.placeholder('float', [None, 1])
		weight_regular = tf.add_n([REGULARIZATION_FACTOR * tf.nn.l2_loss(var) for var in self.net])
		self.loss = tf.reduce_mean(tf.square(self.q_input - self.q_value_output)) + weight_regular
		self.opt = tf.train.AdamOptimizer(LR).minimize(self.loss)
		
		self.action_gradients = tf.gradients(self.q_value_output, self.action_input)
		
# 		initialize network
		self.sess.run(tf.initialize_all_variables())
		self.do_target_update()
	
	def do_target_update(self):
		self.sess.run(self.target_update)
	
	def train_q(self, q_batch, state_batch, action_batch):
		self.update_time += 1
		self.sess.run(self.opt, feed_dict = {
			self.q_input: q_batch,
			self.state_input: state_batch,
			self.action_input: action_batch
		})
	
	def calc_target_q(self, state_batch, action_batch):
		return self.sess.run(self.target_q_value_output, feed_dict = {
			self.target_state_input: state_batch,
			self.target_action_input: action_batch
			})
	
	def calc_q(self, state_batch, action_batch):
		return self.sess.run(self.q_value_output, feed_dict = {
			self.state_input: state_batch,
			self.action_input: action_batch
			})
	
	def calc_gradients(self, state_batch, action_batch):
		return self.sess.run(self.action_gradients, feed_dict = {
			self.state_input: state_batch,
			self.action_input: action_batch
			})[0]
	
