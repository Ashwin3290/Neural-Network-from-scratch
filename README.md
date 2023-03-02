# Neural-network-from-scratch

Introduction
---
Welcome to my project where I have implemented a neural network from scratch using only NumPy library. The purpose of this project was to gain a deeper understanding of the mathematical functions that drive neural networks and learn how they can be implemented in code.

Through this project, I have studied the concepts of forward and backward propagation, weight initialization, activation functions, loss functions, and optimization algorithms. I have implemented each of these concepts using NumPy functions to create a functioning neural network.

In addition to implementing the network, I have also experimented with various hyperparameters such as the learning rate, number of hidden layers, and number of neurons per layer to understand their impact on the network's performance. I have also used different datasets to test the robustness of the network and to gain insights into its strengths and weaknesses.

About the project
---
This is a neural network model that recognizes hand-written digits. The code reads a CSV file containing a dataset of images of hand-written digits. It then splits the dataset into training and development sets, and trains a neural network model using gradient descent algorithm. The neural network has an input layer of 784 neurons, two hidden layers with 10 neurons each, and an output layer of 10 neurons, which corresponds to the 10 possible digits (0-9).

The model is trained using the forward propagation and backward propagation functions. The forward propagation function computes the output of each layer using the weights and bias, and applies the activation function (ReLU for the hidden layers and softmax for the output layer). The backward propagation function computes the gradients of the cost function with respect to the weights and bias using the chain rule. The weights and bias are then updated using the gradients and the learning rate.

The early stopping technique is used to prevent overfitting. If the development set accuracy does not improve for a certain number of iterations, the training is stopped to prevent the model from overfitting the training data. The model's accuracy is measured using the accuracy function, which computes the percentage of correctly predicted digits in the dataset.

The make_predictions function is used to make predictions on new data using the trained model. The user is also given the option to save the parameters of the trained model using the pickle module.

Math used
---

W1,W2,b1,b2 are weights and biases 
These are random inititalized 

**For forward propagation**

Z1 = W1*X + b1

A1 = g1(Z1)

Z2 = W2A1 + b2

A2 = g2(Z2)


Z1,Z2 are the pre-activation values of the hidden and output layers.

A1,A2 are the activation values of the hidden and output layers.

g1,g2 are the activation functions of the hidden and output layers.


With context to the project 
g1 is ReLU and is given by g1(x)=max(0,x)

g2 is softmax and is given by g2(x)=e*x/∑ez


**For back propagation**

dZ2 = A2 - Y

dW2 = 1/m*dZ2.A1T

db2 = 1/m*∑dZ2

dZ1 = W2T *dZ2 * g'(Z1)

dW1 = 1/m*dZ1.XT

db1 = 1/m*∑dZ1


dZ1,dZ2 are the gradients of the cost function with respect to the pre-activation values of the output and hidden layers.

dW1,dW2 are the gradients of the cost function with respect to the weights of the output and hidden layers.

db1,db2 are the gradients of the cost function with respect to the biases of the output and hidden layers.

g'1 is the derivative of the activation function of the hidden layer(which is ReLU here).

m is the number of examples in the training set.


**Updating paramaters**


W1=W1-α*dW1

b1=b1- α*db1

W2=W2-α*dW2

B2=b2- α*db2


Requirements
---

```
pip install -r requirements.txt
```
---

**Feel free to point out issue or improvements with the code**
