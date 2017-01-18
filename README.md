algorithm experiment about Reinforcement Learning Value Function approximation refers to ./vf files Policy Gradient refers to ./pg files Deterministic Policy Gradient refers to ./dpg files

VF: it supports to Q function approximation using linear function under cartpole-v0 experiment of gym

PG/pg:
	It supports to Cartpole-v0 of OpenAI gym using Policy Gradient Reinforcement Learning for discrete action space.
	The game Cartpole-v0 of OpenAI gym refers to 'https://https://gym.openai.com/' & 'https://github.com/openai/gym'.
	The algorithm refers to: linear function approximation for Policy-Function, see 'Reinforcement Learning: an Antroduction' Chapter 13 Campatible Q-function, see 'Policy Gradient Methods for Reinforcement Learning with Function Approximation' Experience Replay, see 'Playing Atari with Deep Reinforcement Learning' 
	The codes refers to: experiment.py includes training and testing policy.py includes definition of policy class. ExperienceReply.py inlcudes definition of experience reply. algorithm.py includes training a policy on game cartpole-v0

PG/dpg:
	It supports to Deterministic Policy Gradient (DPG) method on MountainCar-continuous game of gym.
	The algorithm refers 1. paper: Deterministic Policy Gradient Algorithms; 2. ATA blog: http://www.atatech.org/articles/66098 except that the output is compressed to (-1, 1) by tanh function due to the experiment environment

PG/ddpg:
	It supports to ddpg (Deep Deterministic Policy Gradient) method using tensorflow under MoutainCarContinuous of gym.
	The method refers to the paper: Continuous control with deep reiforcement learning (https://arxiv.org/abs/1509.02971).
