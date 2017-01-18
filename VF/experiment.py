import sys
import random
import gym
from algorithm import *
from policy import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# initialize random number
random.seed(0)

if __name__ == '__main__':
	gameName = 'CartPole-v0'
	env = gym.make(gameName)
	policy = Policy(env)

	# train a policy
	# policy, rewards = train_Qfunction(env, policy, policy_method = 'epsilon-greedy', num_iter = 10000, num_episode = 10000, update_method = 'Qlearning', gamma = 1.0, alpha = 0.8)

	# train a policy using experience replay
	policy, rewards = train_Qfunction_using_EReply(env, 1000, 100,policy, policy_method = 'epsilon-greedy', num_iter = 10000, num_episode = 10000, update_method = 'Qlearning', gamma = 1.0, alpha = 0.8)
	

	plt.plot(rewards)
	plt.xlabel('times')  
	plt.ylabel('Qvalue')  
	plt.show()

	# print('training done, testing starts')

	# # start game environment

	env.monitor.start('../result/' + gameName + 'experiment', force = True)
	for i in xrange(3):
		state = env.reset()
		for iter2 in range(2000):
			env.render()
			action = policy.get_action_according_policy(state, policy_method = 'greedy')
			state, reward, isterminal, info = env.step(action)
			# print reward
			if isterminal:
				print('Episode {0} is terminal at {1} timesteps.'.format(i, iter2))
				break

	env.monitor.close()


