#!/usr/bin/python
import sklearn.datasets
import sys

if (len(sys.argv) > 1):
	num_moons = int(sys.argv[1])
else:
	num_moons = 1000

if (len(sys.argv) > 2):
	output = sys.argv[2]
else:
	output = 'two_moons'

data = sklearn.datasets.make_moons(num_moons)
f = open(output, 'w')
for i in data[0]:
	print(i[0], i[1], sep=' ', end='\n', file = f)
