# Reinforcement Learning models
This is a series of models I worked on while studying Reinforcement Learning.

Resource study:
* [Sutton and Barto's Reinforcement Learning book](http://incompleteideas.net/book/the-book-2nd.html)
* [David Silver lectures](https://www.davidsilver.uk/teaching/)
* [AI-Core Reinforcement Learning course](https://theaicore.com/)
* [Andrej Karpathy - Deep Reinforcement Learning](https://karpathy.github.io/2016/05/31/rl/)

## Value-Based
**DQN-Atari-Breakout**
This is based on the classic [DeepMind Paper](https://arxiv.org/pdf/1312.5602v1.pdf) which has a [very popular video of the Breakout game](https://www.youtube.com/watch?v=TmPfTpjtdgg). 
This video was my biggest inspiration to study A.I.

The model employs a DQN algorithm with 2 Convolutional Neural Networks (Fixed-Q Learning) to
approximate a value function to learn to play the game. It also utilize the Experience Replay technique.

The method used was TD (Temporal-Difference) Learning.
A trained agent can be found in the folder agents.
![VB_breakout](https://github.com/elisiojsj/Reinforcement-Learning/blob/master/Value-Based/gym-results-breakout/VB_breakout.gif)

**DQN-CartPole**
Implementation of Deep Q Network to solve [CartPole](https://gym.openai.com/envs/CartPole-v1/) environment with a simple Neural Network, TD Learning, Q-target and Experience Replay. 

**Q-Learning-CartPole**
Q Learning implementation to solve [CartPole](https://gym.openai.com/envs/CartPole-v1/) environment with a simple Neural Network.
A trained agent can be found in the folder agents.

**Q-Learning-MountainCar**
Q Learning implementation to solve [MountainCar](https://gym.openai.com/envs/MountainCar-v0/) environment with a simple Neural Network.
A trained agent can be found in the folder agents.

**Tabular-Q-Learning**
A tabular classical Q-Learning was used to implement the [Taxi-v3]('https://gym.openai.com/envs/Taxi-v3/') environment. As it's a simpler and discrete game the classical model could easily solve it.

**Tabular-QLearning-SARSA-MountainCar**
An interesting approach was used to make it possible to apply a classical tabular Q-Learning on a more complex continuous environment like [MountainCar](https://gym.openai.com/envs/MountainCar-v0/) using numpy functions of digitize and linspace.
Furthermore, it counts with the implementation of two control methods for TD which are SARSA and Q-Learning.

## Policy-Based
**PolicyGradients-Lunar_Lander**
Implementation of REINFORCE, a Monte Carlo policy-gradient algorithm in the [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/).
A trained agent can be found in the folder agents.

**PolicyGradients-Acrobot** 
Implementation of REINFORCE, a Monte Carlo policy-gradient algorithm in the [Acrobot-v1](https://gym.openai.com/envs/Acrobot-v1/).


## Actor-Critic
**ActorCritic-Lunar_Lander**
This is an implementation of the interesting Actor-Critic algorithm which is roughly a way to take the best of the techniques of Policy-Based and Value-Based approaches. This one was implemented in the [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) environment.
A trained agent can be found in the folder agents. And recordings can be found in the folder recordings.
![AC_lunarlander](https://github.com/elisiojsj/Reinforcement-Learning/blob/master/Actor-Critic/recordings-AC-lunarlander/AC_lunarlander.gif)

## Extra
[Playing a Gym Atari game in the jupyter notebook](https://braraki.github.io/research/2018/06/15/play-openai-gym-games/)

## Software version
* conda 4.8.2
* python 3.7.4
* ptorch 1.4.0
* open-ai gym 0.17.1
