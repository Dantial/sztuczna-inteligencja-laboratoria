import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
print(tf.__version__)

observations=1000000
xs = np.random.uniform(low=-10,high=10, size=(observations,1))
zs = np.random.uniform(low=-10,high=10, size=(observations,1))
inputs=np.column_stack((xs,zs))
print(inputs.shape)

noise = np.random.uniform(low=1,high=1, size=(observations,1))
targets = 13*xs + 7*zs - 12
np.savez('TF_dataset', inputs=inputs, targets=targets)
print(targets.shape)

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
targets = targets.reshape(observations,)
xs = xs.reshape(observations,)
zs = zs.reshape(observations,)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs,zs,targets)
ax.set_xlabel('xs')
ax.set_ylabel('zs')
ax.set_zlabel('Targets')
ax.view_init(azim=100)
plt.show()

init_range = 0.1
weights = np.random.uniform(low=-init_range,high=init_range, size=(2,1))
biases = np. random.uniform(low=-init_range,high=init_range, size=1)
print(weights,biases)

targets = targets.reshape(observations,1)
eta = 1
for i in range (100):
  outputs = np.dot(inputs, weights) + biases
  deltas = outputs - targets

  loss = np.sum(deltas ** 2)/2/observations
  print(loss)

  deltas_scaled = deltas/observations
  weights = weights - eta * np.dot(inputs.T, deltas_scaled)
  biases = biases - eta * np.sum(deltas_scaled)

print (weights, biases)

plt.plot(outputs, targets, color='orange')
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()
