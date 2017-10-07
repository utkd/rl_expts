import sys
import gym
import numpy as np
from matplotlib import pyplot as plt

NUM_EPISODES = 500
GAMMA = 0.99
EPSILON = 0.99
ALPHA = 0.9

MEM_SIZE = 50000
MEM_NEEDED = 500
NEIGHBORS = 250

mstates = np.zeros((MEM_SIZE, 4))
mr = {}
ma = {}
mv = {}

episode_start_idx = 0
last_buf_idx = 0

env = gym.make('CartPole-v0')

total_rewards = []
for episode_number in xrange(NUM_EPISODES):
	obs = env.reset()
	reward = 0
	total_reward = 0

	done = False
	while not done:
		if last_buf_idx > MEM_NEEDED and np.random.rand() > EPSILON:
			# select an action
			distances = np.sum((mstates[:episode_start_idx] - obs)**2, axis=1)
			indices = np.argsort(distances)
			indices = indices[:min(len(indices), 500)]	

			# select the best action
			actions = {}
			counts = {}
			for i in indices:
				v = mv[i]
				a = ma[i]
				vnew = actions.get(a, 0) + v
				actions[a] = vnew
				counts[a] = counts.get(a, 0) + 1

			for a in actions.keys():
				actions[a] = actions[a] / counts[a]

			value_action_pairs = [(y,x) for x,y in actions.iteritems()]
			value_action_pairs.sort(reverse=True)
			action = value_action_pairs[0][1]

		else:
			# select random action
			action = np.random.randint(2)

		if last_buf_idx < MEM_SIZE:
			mstates[last_buf_idx] = obs
			ma[last_buf_idx] = action

			if last_buf_idx > 0:
				mr[last_buf_idx - 1] = reward
				mv[last_buf_idx - 1] = 0

		last_buf_idx += 1

		bestaction = action
		obs, reward, done, info = env.step(bestaction)
		total_reward += reward
		if done:
			v = 0
			for t in reversed(xrange(episode_start_idx, last_buf_idx)):
				v = GAMMA * v + mr.get(t, 0)
				mv[t] = v

			episode_start_idx = last_buf_idx
			EPSILON = EPSILON * 0.98

	print "Episode:", episode_number, "Reward:", total_reward, "Epsilon:", EPSILON
	total_rewards.append(total_reward)

plt.plot(total_rewards)
plt.show()