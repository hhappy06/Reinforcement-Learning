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
	gameName = 'CartPole-v0'
	env = gym.make(gameName)
	policy = Policy(env)

	# # train a policy
	policy, rewards,  timesteps = train_Qfunction(env, policy, buffer_size = 100, batch_size = 50, num_iter = 2000, num_episode = 1000, gamma = 1.0, lr_policy = 0.3, lr_qvalue = 0.3)

	plt.figure()
	p1 = plt.subplot(211)
	p2 = plt.subplot(212)

	p1.plot(rewards)
	p1.set_xlabel('episode')  
	p1.set_ylabel('reward esitmation')

	p2.plot(timesteps)
	p2.set_xlabel('episode')  
	p2.set_ylabel('time-steps')

	# # start game environment

	env.monitor.start('../result/' + gameName + 'experiment', force = True)
	for i in xrange(3):
		state = env.reset()
		for iter2 in range(2000):
			env.render()
			action = policy.get_action_according_policy(state)
			state, reward, isterminal, info = env.step(action)
			# print reward
			if isterminal:
				print('Episode {0} is terminal at {1} timesteps.'.format(i, iter2))
				break

	env.monitor.close()

	plt.show()

