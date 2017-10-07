import gym
import numpy as np
import theano
import theano.tensor as T

import lasagne

NN_NUM_HIDDEN = 20
NN_LEARNING_RATE = 0.05
NUM_EPISODES = 500
GAMMA = 0.99
BATCH_SIZE = 500

def build_mlp(input_var=None):
	l_in = lasagne.layers.InputLayer(shape=(None,4), input_var=input_var)
	l_hid = lasagne.layers.DenseLayer(l_in, num_units=NN_NUM_HIDDEN, nonlinearity=lasagne.nonlinearities.rectify)
	l_out = lasagne.layers.DenseLayer(l_hid, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
	return l_out

def discount_rewards(all_rewards):
	disc_rewards = np.zeros_like(all_rewards)
	disc_so_far = 0.
	for i in reversed(range(len(all_rewards))):
		disc_so_far = disc_so_far * GAMMA + all_rewards[i]
		disc_rewards[i] = disc_so_far
	return disc_rewards

print "Building network ..."
input_var = T.dmatrix('input')
target_var = T.ivector('targets')
adv_var = T.dvector('advantage')
network = build_mlp(input_var)

action_probs = lasagne.layers.get_output(network)
N = input_var.shape[0]
loss = -T.log(action_probs[T.arange(N), target_var]).dot(adv_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.rmsprop(loss, params, learning_rate=NN_LEARNING_RATE)

print "Compiling functions ..."
train_fn = theano.function([input_var, target_var, adv_var], loss, updates=updates)
eval_fn = theano.function([input_var], action_probs)

env = gym.make('CartPole-v0')

total_rewards = []

batch_advantages = []
batch_states = []
batch_actions = []
batch_rewards = []

print "Training ..."

for episode_number in xrange(NUM_EPISODES):
	observation = env.reset()
	done = False
	episode_rewards = []

	while not done:
		curr_state = np.reshape(observation, (-1,4))
		batch_states.append(curr_state)
		probs = eval_fn(curr_state)
		action = np.random.choice([0,1], p=probs[0])

		observation, reward, done, info = env.step(action)
		episode_rewards.append(reward)
		batch_actions.append(action)

		# Manually terminate the episode after 200 time steps
		if len(episode_rewards) > 200:
			done = True
		
	total_rewards.append(np.sum(episode_rewards))
	discounted_rewards = discount_rewards(episode_rewards)
	batch_rewards.append(discounted_rewards)
	
	if len(batch_states) > BATCH_SIZE:
		# Advantage computation that estimates a value function from the current batch
		# Based on http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/pg-startercode.py
		maxlen = max([len(r) for r in batch_rewards])
		padded_rets = [np.concatenate([ret, np.zeros(maxlen-len(ret))]) for ret in batch_rewards]
		baseline = np.mean(padded_rets, axis=0)
		advs = [ret - baseline[:len(ret)] for ret in batch_rewards]
		
		batch_advantages = np.concatenate(advs)
		batch_states = np.array(batch_states).reshape(-1,4)
		currloss = train_fn(batch_states, batch_actions, batch_advantages)
		
		batch_rewards = []
		batch_states = []
		batch_actions = []
		batch_advantages = []

		print "Episode:", episode_number, "Total Reward:", np.sum(episode_rewards)
