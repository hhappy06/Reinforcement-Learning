import sys
import random
import gym
from algorithm import *
from policy import *

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt

# initialize random number
random.seed(0)

if __name__ == '__main__':
	gameName = 'MountainCarContinuous-v0'
	env = gym.make(gameName)
	policy = Policy(env)
	
	# print env.action_space.low[0]
	# print env.action_space.high[0]
	# # train a policy
	policy, rewards,  timesteps = train_Qfunction1(env, policy, buffer_size = 8000, batch_size = 200, num_iter = 1000, num_episode = 3000, gamma = 0.99, lr_policy = 0.005, lr_qfunction = 0.03, lr_qvalue = 0.03)

	plt.figure()
	p1 = plt.subplot(211)
	p2 = plt.subplot(212)

	p1.plot(rewards)
	p1.set_xlabel('episode')  
	p1.set_ylabel('reward estimation')

	p2.plot(timesteps)
	p2.set_xlabel('episode')  
	p2.set_ylabel('time-steps')

	# # start game environment

	max_reward = -100;
	env.monitor.start('../result/' + gameName + 'experiment', force = True)
	for i in xrange(3):
		state = env.reset()
		reward_episode = 0
		for iter2 in range(2000):
			env.render()
			action = policy.get_action_from_policy(state)
			state, reward, isterminal, info = env.step(action)
			reward_episode += reward
			if isterminal:
				print('Episode {0} is terminal at {1} timesteps, reward: {2}.'.format(i, iter2, reward_episode))
				break

	env.monitor.close()

	plt.show()

