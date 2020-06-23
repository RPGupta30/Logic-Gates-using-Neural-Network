#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#Input datasets
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[1],[0],[0],[0]])

def XNOR():
    epochs = 50000
    lr = 0.1
    inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1

    #Random weights and bias initialization
    hw = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
    hb =np.random.uniform(size=(1,hiddenLayerNeurons))
    ow = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
    ob = np.random.uniform(size=(1,outputLayerNeurons))


    #Training algorithm
    for _ in range(epochs):

        # Forward Propagation
        hidden_layer_activation = np.dot(inputs,hw)
        hidden_layer_activation += hb
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output,ow)
        output_layer_activation += ob
        predicted_output = sigmoid(output_layer_activation)

        # Backpropagation
        error = expected_output - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)
    
        error_hidden_layer = d_predicted_output.dot(ow.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Updating Weights and Biases
        ow  += hidden_layer_output.T.dot(d_predicted_output) * lr
        ob  += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
        hw  += inputs.T.dot(d_hidden_layer) * lr
        hb  += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr
    return predicted_output

print('Output of the XNOR Gate',*expected_output)
print("Predicted output from neural network after given epochs: ",*XNOR())
print('\n')


# In[ ]:




