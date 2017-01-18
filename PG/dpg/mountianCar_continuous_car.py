import numpy as np
import gym
import random
random.seed(0)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

gameName = 'MountainCarContinuous-v0'
env = gym.make(gameName)
parameters = np.random.rand(2)-1

def run_episode(env, parameters):
    observation = env.reset()
    total_reward = 0
    for _ in xrange(500):
        action = np.dot(parameters, observation)
        action = np.clip(action, -1, 1)
        action = np.array([action])
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
           break
    return total_reward


bestparams = None
bestreward = -100
for _ in xrange(10000):
    parameters = np.random.rand(2)  - 1
    reward = run_episode(env,parameters)
    if reward > bestreward:
        bestreward = reward
        bestparams = parameters
    # print '{0}, {1}'.format(reward, bestreward)

print 'best parameters:{0}; reward: {1}'.format(bestparams, bestreward)

rewards = []
env.monitor.start('../result/' + gameName + 'experiment', force = True)
for i in xrange(50):
    totalreward = 0
    observation = env.reset()
    for iter2 in range(1000):
        env.render()
        action = np.dot(parameters, observation)
        action = np.clip(action, -1, 1)
        action = np.array([action])
        observation, reward, done, info = env.step(action)
        totalreward += reward
        print '{0},{1}'.format(reward, iter2)
        if done:
            break
        rewards.append(totalreward)

env.monitor.close()
