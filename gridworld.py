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
NUM_EPOCHS	= 20
NOISE		= 0.10
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
DISCOUNT	= 0.00
LRATE		= 0.25
EPSILON_CNT	= 10
EPSILON_RNG	= [float(i)/EPSILON_CNT for i in range(EPSILON_CNT+1)]
INIT_VALUE	= 1.0

# Stats parameters
NUM_RUNS	= 100000

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

	def update(self, state, action, reward):
		val = self.values[self.state[1]][self.state[0]][action]
		target = reward + DISCOUNT*max(self.values[state[1]][state[0]].values())
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

		# State transition
		if rn.random() >= self.noise:
			state = (
						state[0] + action[0],
						state[1] + action[1],
					)
		else:
			state = (
						state[0] + action[0] + rn.randint(-1, 1),
						state[1] + action[1] + rn.randint(-1, 1),
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

world = World(goal=GOAL, noise=NOISE)

stats = OrderedDict([(e, {'mean': [],'stdev': []}) for e in EPSILON_RNG])

for eps in EPSILON_RNG:

	print "Epsilon %3.1f..." % eps

	# Reward history
	data = [[] for i in range(NUM_EPOCHS)]

	print "Learing... "
	for r in tqdm(range(NUM_RUNS)):

		# Agent Init
		agent = Agent(state=AGENT_INIT, init_value=INIT_VALUE, learning_rate=LRATE)

		# Learning process
		for e in range(NUM_EPOCHS):

			# Agent takes an action
			action = agent.greedy(epsilon=eps)

			# World update
			state, reward = world.pull(agent.state, action)

			# Agent learns
			agent.update(state=state, action=action, reward=reward)

			# Reward history update
			data[e].append(reward)


	print "Computing stats... "
	for flt in tqdm([map(float,d) for d in data]):
		stats[eps]['mean'].append(stat.mean(flt))
		stats[eps]['stdev'].append(1.96*stat.stdev(flt)/(NUM_RUNS**0.5))

# Rewards
plt.title('Rewards Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.xlim([0, NUM_EPOCHS-1]) 
plt.tight_layout()
for eps in stats:
	plt.errorbar(range(NUM_EPOCHS), stats[eps]['mean'], yerr=stats[eps]['stdev'], errorevery=NUM_EPOCHS/10)
plt.legend(["Epsilon: %3.1f" % e for e in EPSILON_RNG], loc='best')
plt.grid(which='major', axis='y')
plt.savefig('plot_%4.2f.eps' % DISCOUNT)
plt.savefig('plot_%4.2f.png' % DISCOUNT)
plt.show()
