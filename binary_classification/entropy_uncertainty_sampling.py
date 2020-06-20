#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 08:23:04 2020

@author: jmamath
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import active_learning, plot_decision_boundary, sample_highest_entropy, logistic_regression


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
plt.xlim(-5,5)
plt.ylim(-4,4)
plt.legend()

# Now, we duplicate in two the dataset to use in each procedure
# random sampling with data_rs, and uncertainty sampling with data_us
data_us = np.copy(x)

####################  ACTIVE LEARNING #################### 

evaluation_us, weights_us, training_us  = active_learning(data_us, n_iter=10, n_sample=10, epochs=500, acquisition_function=sample_highest_entropy)
line_us_x, line_us_y = plot_decision_boundary(weights_us)

# Plot the decision boundary learned with the uniformly sampled data
id0 = training_us[:,2] == 0
id1 = training_us[:,2] == 1
rs_id0 = training_us[id0]
rs_id1 = training_us[id1]

# Start with just the data we trained with
plt.scatter(rs_id0[:,0], rs_id0[:,1], s=30., label='N1')
plt.scatter(rs_id1[:,0], rs_id1[:,1], s=30., label='N2')
plt.plot(line_us_x, line_us_y, label="Uncertainty Sampling")
plt.xlim(-5,5)
plt.ylim(-4,4)
plt.title("Decision Boundary with the data we trained on ")
plt.legend()

# Plot the decition boundary with all the data
plt.scatter(x1[:,0], x1[:,1], s=10., label='N1')
plt.scatter(x2[:,0], x2[:,1], s=10., label='N2')
plt.plot(line_us_x, line_us_y, label="Uncertainty Sampling")
plt.xlim(-5,5)
plt.ylim(-4,4)
plt.title("Decision Boundary with Total Data")
plt.legend()


### Final accuracy on the set
model = logistic_regression()
model.set_weights(weights_us[-1])
acc_us = model.evaluate(data_us[:,:2], data_us[:,2])[1]




