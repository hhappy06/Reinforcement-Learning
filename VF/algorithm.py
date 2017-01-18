# upate policy using different method 
# now support Sarsa and Q-learning

import random
import numpy as np
from ExperienceReply import *

# initialize random nubmer 
random.seed(0)

# only support 'Sarsa' and 'Qlearning'
def update_Qfunction(policy, state_t0, action_t0, rewardt1, state_t1, action_t1, isterminal = False, method = 'Sarsa', gamma = 0.1, alpha = 0.001):
	q_s_a_t0 = policy.get_Qfunction(state_t0, action_t0)
	
	q_s_a_t1 = 0.0
	if method == 'Sarsa' and not isterminal:
		q_s_a_t1 = policy.get_Qfunction(state_t1, action_t1)

	if method == 'Qlearning' and not isterminal:
		action = policy.get_action_according_policy(state_t1, 'greedy')
		q_s_a_t1 = policy.get_Qfunction(state_t1, action)

	q_s_a_t1 = rewardt1 + gamma*q_s_a_t1;
	delt = q_s_a_t0 - q_s_a_t1

	derivative_Qfunction = policy.get_derivative(state_t0, action_t0)

	policy.theta = policy.theta - alpha*delt*derivative_Qfunction

	# avoid implicit return
	return policy

def update_Qfunction_batch(policy, batch, method = 'Sarsa', gamma = 0.1, alpha = 0.001):
	if len(batch) == 0:
		return

	# calculate 
	for item in batch:
		update_Qfunction(policy, item[0], item[1], item[2], item[3], item[4], item[5], method = 'Sarsa', gamma = 0.1, alpha = 0.001)

	return policy

def train_Qfunction(env, policy, policy_method = 'epsilon-greedy',num_iter = 100, num_episode = 10000, update_method = 'Sarsa', gamma = 0.9, alpha = 0.1):
	# initialize model parameter
	for i in xrange(len(policy.theta)):
		policy.theta[i] = 0.1

	epsilon = 0.1
	decay_factor = 0.9
	rewards = []
	for iter1 in xrange(num_iter):

		print('Training episode {0}...'.format(iter1))
		state = env.reset()
		action = policy.get_action_according_policy(state, 'greedy')
		t = 0
		isTermial = False

		sumreward = 0.0
		while isTermial == False and t < num_episode:

			sumreward += policy.get_Qfunction(state, action)

			state1, reward1, isTermial, info = env.step(action)
			action1 = policy.get_action_according_policy(state1, policy_method, epsilon)
			# print action1

			# if isTermial:
			# 	reward1 = -1
			policy = update_Qfunction(policy, state, action, reward1, state1, action1, isTermial, update_method, gamma, alpha)

			state = state1
			action = action1

			t += 1

			# adjust learning rate
			if isTermial:
				print('training over at timesetp {0} due to termial'.format(t))

			if iter1 % 500 ==0 and alpha > 0.01:
				alpha *= decay_factor

			if iter1 % 100 == 0 and epsilon > 0.01:
				epsilon *= decay_factor

			if t > 0:
				sumreward = sumreward / t
			rewards.append(sumreward)

	return policy, rewards

def train_Qfunction_using_EReply(env, buffer_size, batch_size, policy, policy_method = 'epsilon-greedy',\
				num_iter = 100, num_episode = 10000, update_method = 'Sarsa', gamma = 0.9, alpha = 0.1):
	# initialize model parameter
	for i in xrange(len(policy.theta)):
		policy.theta[i] = 0.1

	ER = ExperienceReply(buffer_size)

	epsilon = 0.1
	decay_factor = 0.9
	rewards = []

	update_number = 0
	for iter1 in xrange(num_iter):

		print('Training episode {0}...'.format(iter1))
		state = env.reset()
		action = policy.get_action_according_policy(state, 'greedy')
		t = 0
		isTermial = False

		sumreward = 0.0
		while isTermial == False and t < num_episode:

			sumreward += policy.get_Qfunction(state, action)

			state1, reward1, isTermial, info = env.step(action)
			action1 = policy.get_action_according_policy(state1, policy_method, epsilon)

			# add to experience reply
			ER.add_item(state, action, reward1, state1, action1, isTermial)

			if ER.size() > batch_size and update_number == 0:
				batch = ER.sample_batch(batch_size)
				policy = update_Qfunction_batch(policy, batch, update_method, gamma, alpha)

			state = state1
			action = action1

			t += 1
			update_number = (update_number + 1)%20

			# adjust learning rate
			if isTermial:
				print('training over at timesetp {0} due to termial'.format(t))

			if iter1 % 500 ==0 and alpha > 0.01:
				alpha *= decay_factor

			if iter1 % 100 == 0 and epsilon > 0.01:
				epsilon *= decay_factor

			if t > 0:
				sumreward = sumreward / t
			rewards.append(sumreward)

	return policy, rewards
		