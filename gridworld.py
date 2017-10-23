#!/usr/bin/env python
import operator
import matplotlib.pyplot as plt
import math
import numpy as np
import random as rn
import statistics as stat
from tqdm import tqdm
from collections import OrderedDict

# World parameters
WWIDTH		= 10
WHEIGHT		= 5
NUM_EPOCHS	= 1000
NUM_STEPS	= 20
NOISE		= 0.01
GOAL		= (9,3)
AGENT_INIT	= (4,2)
ACTIONS		= OrderedDict([
			('RIGHT',	( 1, 0)),
			('LEFT',	(-1, 0)),
			('UP',		( 0,-1)),
			('DOWN',	( 0, 1)),
			('STAY',	( 0, 0)),
		])

# Learning parameters
# RANGE_CNT	= 5
# RANGE_RNG	= [0.20 + 0.30*float(i)/RANGE_CNT for i in range(RANGE_CNT+1)]
RANGE_RNG	= [0.00, 0.01, 0.10]
EPSILON 	= 0.05
DISCOUNT	= 0.90
LRATE		= 0.50
INIT_VALUE	= 1.0

# Stats parameters
NUM_RUNS	= 10000

class Agent(object):

	def __init__(self, state=(0,0), init_value=0.0, learning_rate=0.01):
		self.state = state
		self.learning_rate = learning_rate
		self.values = [[OrderedDict([
			(ACTIONS['RIGHT'],	init_value*rn.random()),
			(ACTIONS['LEFT'],	init_value*rn.random()),
			(ACTIONS['UP'],		init_value*rn.random()),
			(ACTIONS['DOWN'],	init_value*rn.random()),
			(ACTIONS['STAY'],	init_value*rn.random()),
		]) for w in range(WWIDTH)] for h in range(WHEIGHT)]
		
	def greedy(self, epsilon=0.00):
		vals = self.values[self.state[1]][self.state[0]]
		if rn.random() >= epsilon:
			return max(vals.iteritems(), key=operator.itemgetter(1))[0]
		return rn.sample(vals.keys(),1)[0]

	def update(self, state, action, reward, discount):
		val = self.values[self.state[1]][self.state[0]][action]
		target = reward + discount*max(self.values[state[1]][state[0]].values())
		delta = self.learning_rate*(target - val)
		self.values[self.state[1]][self.state[0]][action] += delta
		self.state = state

	def __str__(self):
		ret_str = ''
		for a in ACTIONS:
			ret_str += '%s:\n' % a
			for x in agent.values:
				for y in x:
					ret_str += '%6.2f ' % y[ACTIONS[a]]
				ret_str += '\n'
			ret_str += '\n'
		return ret_str

class World(object):

	def __init__(self, goal, noise=0.0):
		self.goal = goal
		self.noise = noise

	def pull(self, state, action):

		# Noisy action
		if rn.random() < self.noise:
			action = rn.sample(ACTIONS.values(),1)[0]

		# State transition
		state = (
					state[0] + action[0],
					state[1] + action[1],
				)

		# World boundaries
		if state[0] >= WWIDTH:
			state = (WWIDTH-1, state[1])
		if state[1] >= WHEIGHT:
			state = (state[0], WHEIGHT-1)
		if state[0] < 0:
			state = (0, state[1])
		if state[1] < 0:
			state = (state[0], 0)

		# Reward
		reward = -1
		if state == self.goal:
			reward = 100

		# Return values
		return state, reward

stats = OrderedDict([(e, {'mean': [],'stdev': []}) for e in RANGE_RNG])

for rng in RANGE_RNG:

	world = World(goal=GOAL, noise=rng)

	print "Range %4.2f..." % rng

	# Reward history
	data = [[] for i in range(NUM_EPOCHS)]

	print "Learing... "
	for r in tqdm(range(NUM_RUNS)):

		# Agent Init
		agent = Agent(state=AGENT_INIT, init_value=INIT_VALUE, learning_rate=LRATE)

		# Learning process
		for e in range(NUM_EPOCHS):

			# Episode
			agent.state = AGENT_INIT
			acc_rwd = 0
			for s in range(NUM_STEPS):

				# Agent takes an action
				action = agent.greedy(epsilon=EPSILON)

				# World update
				state, reward = world.pull(agent.state, action)

				# Agent learns
				agent.update(state=state, action=action, reward=reward, discount=DISCOUNT)

				acc_rwd += reward

			# Reward history update
			data[e].append(acc_rwd)


	print "Computing stats... "
	for flt in tqdm([map(float,d) for d in data]):
		stats[rng]['mean'].append(stat.mean(flt))
		stats[rng]['stdev'].append(1.96*stat.stdev(flt)/(NUM_RUNS**0.5))

# Rewards
plt.title('Rewards Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.xlim([0, NUM_EPOCHS-1]) 
plt.tight_layout()
for rng in stats:
	plt.errorbar(range(NUM_EPOCHS), stats[rng]['mean'], yerr=stats[rng]['stdev'], errorevery=NUM_EPOCHS/10)
plt.legend(["Noise: %4.2f" % e for e in RANGE_RNG], loc='best')
plt.grid(which='major', axis='y')
plt.savefig('plot_%03d.eps' % int(100*DISCOUNT))
plt.savefig('plot_%03d.png' % int(100*DISCOUNT))
plt.show()
