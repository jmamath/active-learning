#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 08:12:22 2020

@author: jmamath
"""

import numpy as np
import keras 

def logistic_regression():    
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1,input_shape=(2,), activation="sigmoid"))  
    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])    
    return model 

# This function is used for all active learning workflows
def sample_random(n_sample, data):
    """
    Sample n_sample from data using a uniform distribution.
    Delete the samples from original data. This function is used to
    provide new data of the active learning procedure from the pool.
    Args:
        n_sample: Int. Number of example to sample
        data: Pool of data.
    Return:
        new_data: the n_sample from data
        data: data - new_data
    """
    len_data = data.shape[0]
    # Sampling from uniform distribution
    indices = np.random.uniform(0,len_data,size = n_sample).astype(np.int16)
    new_data = data[indices,:]
    data = np.delete(data,indices, axis=0)
    return new_data, data

def active_learning(data, n_iter, n_sample, epochs, acquisition_function):
    """
    The training dataset is increased by n_sample example at every iteration.
    Args:
        data: Pool of unseen data
        n_iter: Int. Number of iteration to perform the active learning procedure
        n_sample: Int. Number of sample per iteration
        acquisition_function: acquisition function in the active learning context
    Returns:
        evaluation: List of float. The evaluation of the model trained on data at each iteration
        training_data: Total data we have trained on at the end of the total number of iteration
        weights: parameters of the model at each iteration
    """
    evaluation = []
    weights = []
    for i in range(n_iter):
        print("Iteration: {}".format(i+1))        
        # At the first iteration we sample at random
        if i == 0:
            sampled_data, data = sample_random(n_sample,data)
            training_data = sampled_data   
        print("-------------------------")
        print("Start training")
        model = logistic_regression()        
        model.fit(training_data[:,:2], training_data[:,2], epochs=epochs, verbose=0, shuffle=True)
        print("End training")
        print("-------------------------")
        print("Model Evaluation")
        eval_i = model.evaluate(data[:,:2], data[:,2])[1]
        evaluation.append(eval_i)
        print("Accuracy: {}".format(eval_i))         
        weights.append(model.get_weights())
        # Here we specify the case of random sampling because the function
        # sample_random does not use the model as opposed to all other acquisition functions
        if acquisition_function.__name__ == "sample_random":
            sampled_data, data = sample_random(n_sample,data)
            training_data = np.concatenate([training_data, sampled_data])
        else:
            sampled_data, data = acquisition_function(n_sample, model, data)
            training_data = np.concatenate([training_data, sampled_data])        
        print("---------------------------")
    return evaluation, weights, training_data




# This function is used to plot the decision boundary that a model learnt
# on a binary classification task
def plot_decision_boundary(weight):
    """
    Get the parameters of the last weight and derive the equation of
    the line it draws.
    """
    w_,b_ = weight[-1]    
    line_x = np.linspace(-1,1,100)
    a = -w_[0]/w_[1]
    b = -b_ / w_[1]    
    line_y = a*line_x + b
    return line_x, line_y

def entropy(p):
    '''
    computes the entropy of one binary probability distribution
    '''
    return -(p * np.log2(p) + (1-p) * np.log2(1-p))


def index_high_values(data, n_sample):
    """    
    Get the index of the n_sample highest value in data
    Args:
        data: np array of unordered value
        n_sample: Int. Number of element to draw from the ordered data
    Return:
        idx_high_value: Indexes of the highest values in data
    """
    if not isinstance(data, (np.ndarray,)):
        data = np.array(data)    
    idx_sorted = np.argsort(data)
    idx_high_value = idx_sorted[-n_sample:]
    return idx_high_value
    
def test_index_high_values():
    # Small test to understand what the program does    
    a = np.random.uniform(0,20,10).astype(np.int)
    id_a = index_high_values(a, 5) 
    return a, id_a
        
def sample_highest_entropy(n_sample, model, data):
    '''
    Sample n_sample from data using a uncertainty sampling.
    We select n_sample with the highest entropy that we call: data_with_high_entropy
    Delete the samples from original data.    
    '''
    entropies = entropy(model.predict(data[:,:2])).squeeze()
    id_high_entropies = index_high_values(entropies, n_sample)
    data_with_high_entropy = data[id_high_entropies]
    data = np.delete(data, id_high_entropies, axis=0)
    return data_with_high_entropy, data


def  sample_lowest_margin(n_sample, model, data):
    """
    Disclaimer: this function works only in binary classification.
    First the function computes the margin between two classes
    Then it select n_sample items with the lowest margin.
    Those items are typically those that the model struggle to classify
    because the margin between the predictions is low.
    """
    pred = model.predict(data[:, :2]) # predict unlabeled data     
    pred_others = 1-pred         # compute  the proabilities of the second class  
    margin = np.abs(pred-pred_others) # compute the difference and take the max 
    index = margin.argsort(axis=0)                 #  get the indexes to sort by margin
    data_sorted = np.take_along_axis(data, index , axis=0)  # the new data sorting 
    data_add_training = data_sorted[:n_sample]  # take the first 10 rows, with the lowest margin       
    data_labelled = data_sorted[n_sample:]     # data without the last 10 rows         
    return data_add_training, data_labelled


    
def sample_least_confidence(n_sample, model, data):
    
    pred = model.predict(data[:, :2]).ravel()
    pred_others = 1-pred
    max_pred = np.maximum(pred, pred_others) ## only difference take the maximum
    max_pred = max_pred.reshape(data.shape[0], 1)
    data = np.concatenate((data, max_pred), axis=1)
    index = data[:,3].argsort()
    data = data[index]
    
    
    data_labelled = data[n_sample:, 0:3]
    data_add_training = data[0:n_sample, 0:3]    
    return data_add_training, data_labelled
