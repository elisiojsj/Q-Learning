# Reinforcement Learning models
This is a series of models I worked on while studying Reinforcement Learning.

Resource study:
* [Sutton and Barto's Reinforcement Learning book](http://incompleteideas.net/book/the-book-2nd.html)
* [David Silver lectures](https://www.davidsilver.uk/teaching/)
* [AI-Core Reinforcement Learning course](https://theaicore.com/)
* [Andrej Karpathy - Deep Reinforcement Learning](https://karpathy.github.io/2016/05/31/rl/)

## DQN-Atari-Breakout
This is based on the classic [DeepMind Paper](https://arxiv.org/pdf/1312.5602v1.pdf) which has a [very popular video of the Breakout game](https://www.youtube.com/watch?v=TmPfTpjtdgg). 
This video was my biggest inspiration to study A.I.

The model employs a DQN algorithm with 2 Convolutional Neural Networks (Fixed-Q Learning) to
approximate a value function to learn to play the game. It also utilize the Experience Replay technique.

The method used was TD (Temporal-Difference) Learning.

## PolicyGradients-Lunar_Lander
Implementation of REINFORCE, a Monte Carlo policy-gradient algorithm.


## Extra
[Playing a Gym Atari game in the jupyter notebook](https://braraki.github.io/research/2018/06/15/play-openai-gym-games/)

## Software version
* conda 4.8.2
* python 3.7.4
* ptorch 1.4.0
* open-ai gym 0.17.1
