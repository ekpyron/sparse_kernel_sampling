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
#input, input_labels = sklearn.datasets.make_blobs(1000,centers=2,cluster_std=0.5)

sigma_factor = 0.25
initial_columns = 10
max_columns = 100
tolerance = 0.0001

target_dimensions = 1
num_centers = 2
cutoff_distance = 0.2

n = input.shape[0]

print("START OASIS")

D = squareform(pdist(input, 'sqeuclidean'))
sigma = sigma_factor*numpy.sqrt(D.max())
K = scipy.exp(-D/(sigma**2))

#H = (1/n) * numpy.ones(K.shape)
#K = K - numpy.dot(K,H) - numpy.dot(H,K) + numpy.dot(H,numpy.dot(K,H))

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



# Build KD Tree
lowdimproj=proj[:,0:target_dimensions]
max_dist = lowdimproj.max() - lowdimproj.min()
print("Maximum distance: ", max_dist)
tree = KDTree(lowdimproj)
rho = numpy.zeros(n)
delta = numpy.zeros((n,3))
delta[:,0] = numpy.ones(n)*1000
for i in range(0,n):
	delta[i,1] = i
	if target_dimensions == 1:
		point=lowdimproj[i,0]
	else:
		point=lowdimproj[i,:]
	rho[i]=tree.query_radius(point, r=max_dist*cutoff_distance, count_only=True)

for i in range(0,n):
	for j in Lambda:
		dist = numpy.linalg.norm(lowdimproj[i,:]-lowdimproj[j,:])
		if rho[j] >= rho[i] and dist < delta[i,0]:
			delta[i] = [dist, i, j]

sorteddelta = numpy.sort(delta, axis=0)

centers=lowdimproj[sorteddelta[n-num_centers:n,1].astype(int),:]
calculated_labels=numpy.zeros(n)

for i in range(0,n):
	mindist = numpy.linalg.norm(centers[0]-lowdimproj[i])
	for j in range(0,centers.shape[0]):
		dist = numpy.linalg.norm(centers[j]-lowdimproj[i])
		if dist < mindist:
			mindist = dist
			calculated_labels[i] = j


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
ax.set_title("calculated labels on data")
ax.scatter(input[:,0], input[:,1], color=colors[::-1][calculated_labels.astype(int)].tolist())


plt.show()

