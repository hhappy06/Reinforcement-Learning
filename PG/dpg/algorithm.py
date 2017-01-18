# upate policy using different method 
# now support Sarsa and Q-learning

import random
import numpy as np
from ExperienceReply import *

# initialize random number
random.seed(0)

def train_Qfunction1(env, policy, buffer_size= 5000, batch_size= 100, num_iter=100, num_episode=10000, gamma=0.99, lr_policy=0.3, lr_qfunction=0.3, lr_qvalue=0.3):
	lr_decay = 0.9
	replay = ExperienceReply(buffer_size)
	replay_update_count = 0
	policy_update_count = 0
	
	rewards = []
	timesteps = []
	
	for iter1 in xrange(num_iter):
		
		state = env.reset()
		action = policy.get_action_from_policy(state, True)
		
		t = 0
		isTerminal = False
		nupdate_episode = 0
		
		rewards_episode = 0
		episode_buffer = []
		while isTerminal == False and t < num_episode:
			state1, reward1, isTerminal, info = env.step(action)
			action1 = policy.get_action_from_policy(state1, True)
			episode_buffer.append([state.T, action, reward1, state1.T, action1, isTerminal])
			
			state = state1
			action = action1
			t += 1
			
			rewards_episode += reward1
			replay_update_count += 1
		
		# print 'episode {0} length: {1}'.format(iter1,t)
		# if isTerminal:
		# 	print('training over at time-step {0} due to terminal'.format(t))
			
		discount_reward = 0.0
		for idx in xrange(len(episode_buffer)-1, -1, -1):
			episode_buffer[idx][2] += gamma * discount_reward
			discount_reward = episode_buffer[idx][2]
		
		for idx in xrange(len(episode_buffer)):
			replay.add_item(episode_buffer[idx][0],episode_buffer[idx][1],episode_buffer[idx][2],episode_buffer[idx][3],episode_buffer[idx][4],episode_buffer[idx][5])
			
			if replay.size() > batch_size and idx % 10 == 0:
				batch = replay.sample_batch(batch_size)
				# batch_reward, batch_reward2 = policy.update_policy_batch(batch, gamma, lr_policy * (1.0 - 0.8*t/num_episode), lr_qfunction * (1.0 - 0.5*t/num_episode), lr_qvalue * (1.0 - 0.8*t/num_episode))
				batch_reward, batch_reward2 = policy.update_policy_batch(batch, gamma, lr_policy, lr_qfunction, lr_qvalue)
				
				policy_update_count += 1
				nupdate_episode += 1
				
				# output update info
				print('episode_{0}_length_{1}: update {2} state:{3}; rd:{4}; {5}, {6}, {7}'.format(iter1, t, policy_update_count, episode_buffer[idx][0], batch_reward, batch_reward2, episode_buffer[idx][1], episode_buffer[idx][2]))
				
				if policy_update_count % 3000 == 0:
					if lr_policy > 0.005:
						lr_policy *= lr_decay
					if lr_qvalue > 0.05:
						lr_qvalue *= lr_decay
					if lr_qfunction > 0.05:
						lr_qfunction *= lr_decay
			
		
		if t > 0:
			rewards_episode /= t
		else:
			rewards_episode = 0
		rewards.append(rewards_episode)
		timesteps.append(t)
	
	return policy, rewards, timesteps

def train_Qfunction2(env, policy, buffer_size= 2000, batch_size= 200, num_iter=100, num_episode=10000, gamma=0.99, lr_policy=0.3, lr_qfunction=0.3, lr_qvalue=0.3):
	lr_decay = 0.9
	replay = ExperienceReply(buffer_size)
	replay_update_count = 0
	policy_update_count = 0
	
	rewards = []
	timesteps = []
	
	for iter1 in xrange(num_iter):
		
		state = env.reset()
		action = policy.get_action_from_policy(state, True)
		
		t = 0
		isTerminal = False
		nupdate_episode = 0
		
		rewards_episode = 0
		while isTerminal == False and t < num_episode:
			state1, reward1, isTerminal, info = env.step(action)
			action1 = policy.get_action_from_policy(state1, True)
			replay.add_item(state.T, action, reward1, state1.T, action1, isTerminal)
							
			state = state1
			action = action1
			
			t += 1
			rewards_episode += reward1
			replay_update_count += 1
			
			if replay.size() > batch_size:
				batch = replay.sample_batch(batch_size)
				batch_reward, batch_reward2 = policy.update_policy_batch(batch, gamma, lr_policy, lr_qfunction, lr_qvalue)
				policy_update_count += 1
				
				nupdate_episode += 1
				
				# output update info
				print('episode_{0}_length_{1}: update {2} lr_p: {3}; lr_q: {4}; rd:{5}; {6}, {7}, {8}'.format(iter1, t, policy_update_count, lr_policy, lr_qvalue, batch_reward, batch_reward2, action, reward1))
				
				if policy_update_count % 3000 == 0:
					if lr_policy > 0.005:
						lr_policy *= lr_decay
					if lr_qvalue > 0.01:
						lr_qvalue *= lr_decay
					if lr_qfunction > 0.01:
						lr_qfunction *= lr_decay
			
			if isTerminal:
				print('training over at time-step {0} due to terminal'.format(t))
				
		if t > 0:
			rewards_episode /= t
		else:
			rewards_episode = 0
		rewards.append(rewards_episode)
		timesteps.append(t)
	
	return policy, rewards, timesteps