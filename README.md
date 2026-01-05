Wids Learnings

Week-1 : Learned Linear Regression and Logistic Regression (Already had knowledge about)

Week-2 : Tensorflow
Tensor is a multidimensional array that has ability to be run on gpu
Generalism for scalar vector matrix 
Ops and everything 

Keras : user friendly library to build neural networks
It hides all the math and GPU in it
Each layer:
Takes input numbers


Applies a mathematical operation


Produces output numbers

Working with mnist dataset : train on 60,000 examples and test on 10,000
Greyscale image- one channel(intensity) 28 by 28 pixels
Sequential API
Softmax on output layer
from_logits=true (applies softmax)
Optimizer is algo that tells the Neural Network how to change the weights
model.fit() it makes the model actually learn using the dataset model.compile()  tells the model how it can learn
epoch=one entire run on the dataset
verbose=in keras u can control what u want to print while training
Batch_size = sample size that model processes one at a time before updating weights
Shape is tuple
Functional API
