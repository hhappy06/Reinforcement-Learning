import sys
import random
import gym
from Actor_Critic import ActorCritic
from algorithm import *
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt

# initialize random number
random.seed(0)

if __name__ == '__main__':
	gameName = 'MountainCarContinuous-v0'
	env = gym.make(gameName)
	
	# print env.action_space.low[0]
	# print env.action_space.high[0]
	# # train a policy
	ob_examples = np.array([env.observation_space.sample() for _ in xrange(10000)])
	state_mean = np.mean(ob_examples, axis=0)
	state_std = np.std(ob_examples, axis=0) + 1.0e-6  # avoid 0
	
	agent, episode_length, episode_reward = train_actor_critic(env, buffer_size = 100000, batch_size = 64, num_iter = 2000, num_episode = 5000)
	
	plt.figure()
	p1 = plt.subplot(211)
	p2 = plt.subplot(212)

	p1.plot(episode_reward)
	p1.set_xlabel('episode')
	p1.set_ylabel('total reward')

	p2.plot(episode_length)
	p2.set_xlabel('episode')
	p2.set_ylabel('episode-steps')
	
	# # start game environment
	max_reward = -100;
	env.monitor.start('../result/' + gameName + 'experiment', force=True)
	for i in xrange(3):
		state = env.reset()
		reward_episode = 0
		for iter2 in range(2000):
			env.render()
			state = (state - state_mean) / state_std
			action = agent.action(state)
			state, reward, isterminal, info = env.step(action)
			reward_episode += reward
			if isterminal:
				print('Episode {0} is terminal at {1} timesteps, reward: {2}.'.format(i, iter2, reward_episode))
				break
	
	env.monitor.close()
	
	plt.show()
	
# 	save result
	agent.save_network('network_20161215')
