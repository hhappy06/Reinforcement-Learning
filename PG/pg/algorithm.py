# upate policy using different method 
# now support Sarsa and Q-learning

import random
import numpy as np
from ExperienceReply import *

# initialize random nubmer 
random.seed(0)



def train_Qfunction(env, policy, buffer_size = 500, batch_size = 100, num_iter = 100, num_episode = 10000, gamma = 1.0, lr_policy = 0.3, lr_qvalue = 0.3):

	lr_decay = 0.9
	replay = ExperienceReply(buffer_size)
	replay_update_count = 0
	policy_update_count = 0

	rewards = []
	timesteps = []
	rewards_episode = 0

	for iter1 in xrange(num_iter):

		state = env.reset()
		action = policy.get_action_according_policy(state)
		
		t = 0
		isTermial = False
		nupdate_episode = 0
		episode_buffer = []

		while isTermial == False and t < num_episode:

			state1, reward1, isTermial, info = env.step(action)
			action1 = policy.get_action_according_policy(state1)
			episode_buffer.append((state, int(action), reward1, state1, int(action1), isTermial))

			state = state1
			action = action1
			t += 1

		# modify the reward and add to experience replay
		rewards_episode = 0
		for i in xrange(len(episode_buffer)):
			replay.add_item(episode_buffer[i][0], int(episode_buffer[i][1]), episode_buffer[i][2] + t - i - 1, episode_buffer[i][3], int(episode_buffer[i][4]), episode_buffer[i][5])
			replay_update_count += 1

			if replay.size() > batch_size and replay_update_count % 20 == 0:
				batch = replay.sample_batch(batch_size)
				batch_reward = policy.update_policy_batch(batch, gamma, lr_policy, lr_qvalue)
				policy_update_count += 1

				rewards_episode += batch_reward
				nupdate_episode += 1

				# output update infor
				print('episode_{0}_length_{1}: update {2} lr_p: {3}; lr_q: {4}; rd:{5};'.format(iter1, t, policy_update_count, lr_policy, lr_qvalue, batch_reward))

				if policy_update_count % 200 == 0:
					if lr_policy > 0.02:
							lr_policy *= lr_decay
					if lr_qvalue > 0.02:
							lr_qvalue *= lr_decay
			# if isTermial:
			# 	print('training over at timesetp {0} due to termial'.format(t))
		if nupdate_episode > 0:
			rewards_episode /= nupdate_episode
		else:
			rewards_episode = 0
		rewards.append(rewards_episode)
		timesteps.append(t)

	return policy, rewards, timesteps


		