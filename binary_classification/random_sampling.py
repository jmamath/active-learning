#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 08:21:39 2020

@author: jmamath
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import sample_random, active_learning, plot_decision_boundary, LogisticRegression

#################### INITIAL DATA #################### 
# We start by generating data from two normal distribution on the plane.
x1 = np.random.multivariate_normal(mean=np.array([-2,0]), cov=np.eye(2), size = 200)
x2 = np.random.multivariate_normal(mean=np.array([2,0]), cov=np.eye(2), size = 200)

# We give to each data belonging a label.
y1 = np.zeros([200,1])
y2 = np.ones([200,1])


x1 = np.concatenate([x1,y1], axis=-1)
x2 = np.concatenate([x2,y2], axis=-1)
x = np.concatenate([x1,x2]).astype(np.float32)


# Display the original data
plt.scatter(x1[:,0], x1[:,1], s=10., label='N1')
plt.scatter(x2[:,0], x2[:,1], s=10., label='N2')
plt.legend()

# Now, we duplicate in two the dataset to use in each procedure
# random sampling with data_rs, and uncertainty sampling with data_us
data_rs = np.copy(x)

#################### RANDOM SAMPLING WORKFLOW #################### 

evaluation_rs, weights_rs, training_rs = active_learning(data_rs, n_iter=10, n_sample=10, epochs=500, acquisition_function=sample_random)  

line_rs_x, line_rs_y = plot_decision_boundary(weights_rs)

# Plot the decision boundary learned with the uniformly sampled data
id0 = training_rs[:,2] == 0
id1 = training_rs[:,2] == 1
rs_id0 = training_rs[id0]
rs_id1 = training_rs[id1]

# Start with just the data we trained with
plt.scatter(rs_id0[:,0], rs_id0[:,1], s=30., label='N1')
plt.scatter(rs_id1[:,0], rs_id1[:,1], s=30., label='N2')
plt.plot(line_rs_x, line_rs_y, label="Uncertainty Sampling")
plt.xlim(-5,5)
plt.ylim(-4,4)
plt.title("Decision Boundary with the data we trained on ")
plt.legend()

# Plot the decition boundary with all the data
plt.scatter(x1[:,0], x1[:,1], s=10., label='N1')
plt.scatter(x2[:,0], x2[:,1], s=10., label='N2')
plt.plot(line_rs_x, line_rs_y)
plt.plot(line_rs_x, line_rs_y, label="Random Sampling")
plt.xlim(-5,5)
plt.ylim(-4,4)
plt.title("Decision Boundary with Total Data")
plt.legend()

##### Final test
model = LogisticRegression()
model.set_weights(weights_rs[-1])
acc_rs = model.evaluate(data_rs[:,:2], data_rs[:,2])[1]

