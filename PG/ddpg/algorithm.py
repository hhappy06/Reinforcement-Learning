import numpy as np
import gym

from ExperienceReply import ExperienceReply
from Actor_Critic import ActorCritic

def train_actor_critic(env, buffer_size = 5000, batch_size = 32, num_iter = 1000, num_episode = 1000):
	# normalize state space
	ob_examples = np.array([env.observation_space.sample() for _ in xrange(10000)])
	state_mean = np.mean(ob_examples, axis=0)
	state_std = np.std(ob_examples, axis=0) + 1.0e-6 #avoid 0
	replay = ExperienceReply(buffer_size)
	
	agent = ActorCritic(env.observation_space.shape[0], len(env.action_space.sample()))
	ave_episode = 0;
	episode_length = []
	episode_reward = []
	for inter in xrange(num_iter):
		state = env.reset()
		# normalize state using mean and standard variance
		state = (state - state_mean) / state_std
		
		isTerminal = False
		t = 0
		reward_episode = 0
		state_scope = [0.0, 0.0]
		while not isTerminal and t < num_episode:
			action = agent.action_explore(state)
			state1, reward1, isTerminal, _ = env.step(action)
			
			reward_episode += reward1
			state_scope[0] = min(state_scope[0], state1[0])
			state_scope[1] = max(state_scope[1], state1[0])
			state1 = (state1 - state_mean) / state_std
			
			replay.add_item(state, action, reward1, state1, isTerminal)
			
			if replay.size() > batch_size and t % 5 == 0:
				batch = replay.sample_batch(batch_size)
				# batch = replay.sample_batch(batch_size, sample_method='prob')
				average_reward, average_target, sigma = agent.train(batch)
				
				print 'episode:{0}_{1}; step_{2}, ave_red:{3} ave_tar: {4},[{5}, {6}],l:{7}'.format(inter, t, agent.time_update, average_reward, average_target, state_scope[0], state_scope[1], ave_episode)
			
			state = state1
			t += 1
			
		ave_episode = (ave_episode * inter + t) / (inter + 1)
		# log train info
		episode_length.append(t)
		episode_reward.append(reward_episode)
		
	return agent, episode_length, episode_reward
