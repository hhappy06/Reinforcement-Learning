from collections import deque
import numpy as np
import random
random.seed(0)

class ExperienceReply():
	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		self.count = 0
		self.buffer = deque()
		self.weight = deque()
		self.sum = 0.0

	def add_item(self, state0, action0, reward1, state1, isTerminal = False):
		# item is a tuple 
		item = (state0, action0, reward1, state1, isTerminal)
		weight = self.calc_probobility(reward1)
		
		if self.count < self.buffer_size:
			self.count += 1
		else:
			self.buffer.popleft()
			last_weight = self.weight.popleft()
			self.sum -= last_weight
			
		self.buffer.append(item)
		self.weight.append(weight)
		self.sum += weight
		
	def size(self):
		return self.count
	
	def calc_probobility(self, reward):
		return np.exp(min(reward, 10))
	
	def sample_batch(self, batch_size, sample_method = 'uniform'):
		if self.count < batch_size:
			return random.sample(self.buffer, self.count)
		
		if sample_method == 'uniform':
			return random.sample(self.buffer, batch_size)
		else:
			batch = []
			for i in xrange(batch_size):
				item = self.get_item_from_probability()
				batch.append(item)
			return batch
				
	def get_item_from_probability(self):
		x = random.uniform(0, self.sum)
		cumulative_probability = 0.0
		for idx in xrange(len(self.buffer)):
			cumulative_probability += self.weight[idx]
			if x < cumulative_probability: break
		return self.buffer[idx]
	
	def clear(self):
		self.deque.clear()
		self.count = 0
