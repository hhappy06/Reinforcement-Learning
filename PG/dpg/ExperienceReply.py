from collections import deque
import numpy as np
import random
random.seed(0)

class ExperienceReply():
	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		self.ncount = 0
		self.buffer = deque()

	def add_item(self, state0, action0, reward1, state1, action1, isTerminal = False):
		# item is a tuple 
		item = (state0, action0, reward1, state1, action1, isTerminal)
		if self.ncount < self.buffer_size:
			self.buffer.append(item)
			self.ncount += 1
		else:
			self.buffer.popleft()
			self.buffer.append(item)

	def size(self):
		return self.ncount

	def sample_batch(self, batch_size):
		batch = []

		if self.ncount < batch_size:
			batch = random.sample(self.buffer, self.count)
		else:
			batch = random.sample(self.buffer, batch_size)

		return batch

	def clear(self):
		self.deque.clear()
		self.ncount = 0
