#!/usr/bin/python

import csv
import sklearn.datasets
import numpy
import scipy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

#input, input_labels = sklearn.datasets.make_moons(1000,noise=0.01)
input, input_labels = sklearn.datasets.make_circles(1000,factor=0.3,noise=0.05)
#input, input_labels = sklearn.datasets.make_blobs(1000,centers=4,cluster_std=0.75)

sigma_factor = 0.25
initial_columns = 1
max_columns = 100
tolerance = 0.0001

target_dimensions = 1
num_centers = 2
cutoff_distance = 0.05

centering = True

n = input.shape[0]

print("START OASIS")

D = squareform(pdist(input, 'sqeuclidean'))
sigma = sigma_factor*numpy.sqrt(D.max())
K = scipy.exp(-D/(sigma**2))

if centering:
	H = (1/n) * numpy.ones(K.shape)
	K = K - numpy.dot(K,H) - numpy.dot(H,K) + numpy.dot(H,numpy.dot(K,H))

Lambda = numpy.arange(n)
numpy.random.shuffle(Lambda)

k = initial_columns

Lambda = Lambda[:k]

Ctransp = K[Lambda,:]

W = Ctransp[:,Lambda]

Winv = numpy.linalg.pinv(W)

R = numpy.dot(Winv, Ctransp)

d = numpy.diag(K)

while k < max_columns:
	Delta = d - numpy.sum(numpy.multiply(Ctransp, R), axis=0)
	Delta[Lambda] = 0
	i = Delta.argmax()
	err = abs(Delta[i])
	if (err < tolerance):
		break

	s = 1 / Delta[i]
	newrow = K[i,:]

	q = R[:,i]
	sq = s * q
	Winv = Winv + numpy.outer(sq, q)
	Winv = numpy.vstack((Winv,-sq))
	Winv = numpy.column_stack((Winv,numpy.hstack((-sq,[s]))))

	tmp = numpy.dot(q, Ctransp) - newrow
	R = R + numpy.outer(sq, tmp)
	R = numpy.vstack((R,-s*tmp))

	Ctransp = numpy.vstack((Ctransp,newrow))
	
	Lambda = numpy.hstack((Lambda,[i]))
	
	k = k + 1

print("END OASIS")

W = Ctransp[:,Lambda]
U,S,V = numpy.linalg.svd(W, full_matrices=False, compute_uv=1)

Sbar = (n/k) * S
Sbarinv=S
for i in range(0,S.shape[0]):
	if (Sbarinv[i]<=0.00001):
		Sbarinv[i] = 0
	else:
		Sbarinv[i] = 1/Sbarinv[i]
Ubar = numpy.sqrt(k/n) * numpy.dot(Ctransp.transpose(), numpy.dot(U, numpy.diag(Sbarinv)))


proj = Ubar


Kbar = numpy.dot(numpy.dot(Ubar, numpy.diag(Sbar)), Ubar.transpose())

print("|K-K'|: ", numpy.linalg.norm(Kbar-K))



lowdimproj=proj[:,0:target_dimensions]
lowdimproj_landmarks = lowdimproj[Lambda,:]
if centering:
	#tmp = numpy.diag(W)
	#landmark_dists = numpy.sqrt(0.5*(numpy.outer(tmp, numpy.ones(k)) + numpy.outer(numpy.ones(k), tmp) - 2 * W))
	landmark_dists = squareform(pdist(lowdimproj_landmarks, 'euclidean'))
else:
	landmark_dists = numpy.sqrt(1 - W)

max_dist = landmark_dists.max()
print("Maximum distance: ", max_dist)

tree = KDTree(lowdimproj)
rho = numpy.zeros(k)
delta = max_dist*numpy.ones(k)

for i in range(0,k):
	rho[i]=tree.query_radius(lowdimproj[Lambda[i]:Lambda[i]+1,:], r=max_dist*cutoff_distance, count_only=True)

rho_indices = numpy.argsort(rho)[::-1]

delta[rho_indices[0]] = max_dist
for i in range(1,k):
	j = numpy.argmin(landmark_dists[rho_indices[i],rho_indices[0:i]])
	delta[rho_indices[i]] = landmark_dists[rho_indices[i],rho_indices[j]]

indices = numpy.argsort(delta)[::-1]

l=numpy.zeros(k) - 1
for i in range(0,num_centers):
	l[indices[i]] = i

for i in range(1,k):
	if (l[rho_indices[i]] == -1):
		j = numpy.argmin(landmark_dists[rho_indices[i],rho_indices[0:i]])
		l[rho_indices[i]] = l[rho_indices[j]]

calculated_labels=numpy.dot(l*2-1, Ctransp)


landmarks = input[Lambda,:]

landmark_labels = input_labels[Lambda]

extrapolated_labels = numpy.dot(landmark_labels*2-1, Ctransp)




colors = numpy.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = numpy.hstack([colors] * 20)


fig = plt.figure()
ax = fig.add_subplot(2,4,1)
ax.set_title("input data")
ax.scatter(input[:,0], input[:,1], color=colors[input_labels].tolist())

ax = fig.add_subplot(2,4,2)
ax.set_title("landmark points")
ax.scatter(landmarks[:,0], landmarks[:,1], color=colors[landmark_labels].tolist())

ax = fig.add_subplot(2,4,3)
ax.set_title("label extrapolation from landmarks")
ax.scatter(extrapolated_labels[:], numpy.zeros_like(extrapolated_labels[:]))

ax = fig.add_subplot(2,4,4)
ax.set_title("extrapolated labels on data")
rgb =  plt.get_cmap('jet')((extrapolated_labels[:] - extrapolated_labels[:].min(0)) / extrapolated_labels[:].ptp(0))
ax.scatter(input[:,0], input[:,1], color=rgb)


ax = fig.add_subplot(2,4,5)
ax.set_title("projection to two singular values")
ax.scatter(proj[:,0], proj[:,1], color=colors[input_labels].tolist())

ax = fig.add_subplot(2,4,6)
ax.set_title("projection to one singular value")
ax.scatter(proj[:,0], numpy.zeros_like(proj[:,0]), color=colors[input_labels].tolist())

ax = fig.add_subplot(2,4,7)
ax.set_title("calculated labels on landmarks")
ax.scatter(landmarks[:,0], landmarks[:,1], color=colors[::-1][l.astype(int)].tolist())

ax = fig.add_subplot(2,4,8)
ax.set_title("calculated labels on data")
rgb =  plt.get_cmap('jet')((calculated_labels[:] - calculated_labels[:].min(0)) / calculated_labels[:].ptp(0))
ax.scatter(input[:,0], input[:,1], color=rgb)


plt.show()

