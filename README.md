# myrl
Algorithm experiments about Reinforcement Learning including Q-learning (QL), policy gradient (PG) and deterministic policy gradien (DPG) and deep deterministic policy gradient (DDPG)<br>

* Value Function approximation (QL) refers to ./vf files<br> 
* Policy Gradient (PG ) refer to ./pg files<br>
* Deterministic Policy Gradient (DPG and DDPG) refer to ./dpg files<br>

## VF:
* it supports to Q function approximation using linear function under cartpole-v0 experiment of gym

## PG/pg:
* It supports to Cartpole-v0 of OpenAI gym using Policy Gradient Reinforcement Learning for discrete action space.<br>
* The game Cartpole-v0 of OpenAI gym refers to 'https://gym.openai.com/' & 'https://github.com/openai/gym'.<br>
* The algorithm refers to: linear function approximation for Policy-Function, see 'Reinforcement Learning: an Antroduction' Chapter 13 Campatible Q-function, see 'Policy Gradient Methods for Reinforcement Learning with Function Approximation' Experience Replay, see 'Playing Atari with Deep Reinforcement Learning'. <br>
* The codes refers to: experiment.py includes training and testing policy.py includes definition of policy class. ExperienceReply.py inlcudes definition of experience reply. algorithm.py includes training a policy on game cartpole-v0.<br>

## PG/dpg:
* It supports to Deterministic Policy Gradient (DPG) method on MountainCar-continuous game of gym.<br>
* The algorithm refers 1. paper: Deterministic Policy Gradient Algorithms; 2. ATA blog: http://www.atatech.org/articles/66098 except that the output is compressed to (-1, 1) by tanh function due to the experiment environment<br>

## PG/ddpg:
* It supports to ddpg (Deep Deterministic Policy Gradient) method using tensorflow under MoutainCarContinuous of gym.<br>
* The method refers to the paper: Continuous control with deep reiforcement learning (https://arxiv.org/abs/1509.02971).<br>
