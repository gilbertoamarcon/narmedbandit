#!/usr/bin/env python
import operator
import matplotlib.pyplot as plt
import math
import numpy as np
import random as rn
import statistics as stat
from tqdm import tqdm
from collections import OrderedDict

NUM_RUNS	= 10000
NUM_EPOCHS	= 100

class Agent(object):

	def __init__(self, init_value=1.0, learning_rate=0.01):
		self.values = OrderedDict([
			('A1', init_value),
			('A2', init_value),
			('A3', init_value),
			('A4', init_value),
			('A5', init_value),
		])
		self.learning_rate = learning_rate

	def greedy(self, epsilon=0.00):
		if rn.random() >= epsilon:
			return max(self.values.iteritems(), key=operator.itemgetter(1))[0]
		return rn.sample(self.values.keys(),1)[0]

	def update(self, action, reward, epoch):
		self.values[action] += self.learning_rate*(reward - self.values[action])
		return self.values

class NBandit(object):

	def __init__(self ,actions={}):
		self.actions = actions

	def pull(self, action):
		return np.random.normal(self.actions[action]['mean'], self.actions[action]['variance']**0.5, 1)[0]

actions = OrderedDict([
	('A1', {'mean': 1.00, 'variance':  5.00}),
	('A2', {'mean': 1.50, 'variance':  1.00}),
	('A3', {'mean': 2.00, 'variance':  1.00}),
	('A4', {'mean': 2.00, 'variance':  2.00}),
	('A5', {'mean': 1.75, 'variance': 10.00}),
])
nbandit = NBandit(actions=actions)

# choices vector init
choices = OrderedDict([
			('A1', [0.0]*NUM_RUNS),
			('A2', [0.0]*NUM_RUNS),
			('A3', [0.0]*NUM_RUNS),
			('A4', [0.0]*NUM_RUNS),
			('A5', [0.0]*NUM_RUNS),
		])

# data vector init
data = []
for i in range(NUM_EPOCHS):
	data.append(OrderedDict([
			('reward', []),
			('A1', []),
			('A2', []),
			('A3', []),
			('A4', []),
			('A5', []),
		]))

print "Learing... "
for r in tqdm(range(NUM_RUNS)):

	# Agent Init
	agent = Agent(init_value=2.0, learning_rate=0.25)

	# Learning process
	for e in range(NUM_EPOCHS):
		# if e < 0.5*NUM_EPOCHS:
		# 	action = agent.greedy(epsilon=0.5*(1.0+math.cos(2*math.pi*e/NUM_EPOCHS)))
		# else:
		action = agent.greedy(epsilon=1.0)
		reward = nbandit.pull(action)
		values = agent.update(action=action, reward=reward, epoch=e)
		choices[action][r] += 1
		data[e]['reward'].append(reward)
		for v in values:
			data[e][v].append(values[v])

print "Computing stats... "
stats = OrderedDict([
			('reward', ([],[])),
			('A1', ([],[])),
			('A2', ([],[])),
			('A3', ([],[])),
			('A4', ([],[])),
			('A5', ([],[])),
		])
for s in tqdm(data):
	for k in stats:
		stats[k][0].append(stat.mean(map(float,s[k])))
		stats[k][1].append(1.96*stat.stdev(map(float,s[k]))/(NUM_RUNS**0.5))


# Values
plt.subplot2grid((1,4), (0,0), colspan=3)
plt.title('Rewards and action values')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.xlim([0, NUM_EPOCHS]) 
plt.tight_layout()
for k in stats:
	plt.errorbar(range(NUM_EPOCHS), stats[k][0], yerr=stats[k][1], errorevery=10)
plt.legend(stats.keys(), loc='best')
plt.grid(which='major', axis='y')

# Actions
plt.subplot2grid((1,4), (0,3))
plt.title('Action Choice')
plt.xlabel('Action')
plt.ylabel('Frequency')
plt.tight_layout()
ameans = []
aerror = []
for k in choices:
	ameans.append(stat.mean(choices[k]))
	aerror.append(1.96*stat.stdev(choices[k])/(NUM_RUNS**0.5))
y_pos = np.arange(len(choices))
plt.bar(y_pos, ameans, yerr=aerror, align='center', ecolor='k')
plt.xticks(y_pos, choices.keys())
plt.show()

