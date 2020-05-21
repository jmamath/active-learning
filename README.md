# Active-learning
The plan here is to reproduct the results from "Gal, Y., Islam, R. and Ghahramani, Z., 2017, August. Deep bayesian active learning with image data. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 (pp. 1183-1192). JMLR. org." on the MNIST dataset

## 0. How to contribute
Each active learning with take place with each acquisition function on dedicated script to keep things clean. If the script requires many functions, they will be put in utils.py
Note that experiment_1.py is the first implementation of the code, it is messy. 

## 1. Warm up
The plan is the following:
First we are going to reproduce basic acquisition functions on a synthetic binary classification example following Figure 2 in "Settles, B., 2009. Active learning literature survey. University of Wisconsin-Madison Department of Computer Sciences.".
Those are:
* Random sampling
* Entropy
* Least confident
* Margin sampling

Then we are going to move to MNIST

## 2. Attacking the MNIST dataset
At this point we are going to start with random sampling and entropy with Lenet-5 on MNIST.
Then, we will implement 
* BALD, 
* Mean standard deviation
* Variation Ratios 

## 3. Going Bayesian
Then we will implement Bayesian Lenet-5, and compare it to Lenet-5 with all the acquisition functions



